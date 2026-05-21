{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE BangPatterns #-}

module Bow where

import Codec.Binary.UTF8.String (encode)
import GHC.Generics
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as C
import Data.Word (Word8)
import qualified Data.Map.Strict as M
import Data.List (nub, foldl')
import Data.Char (toLower, isAlphaNum)
import Data.Int (Int64)
import Control.Monad (when, foldM)
import System.IO (hFlush, stdout)

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', meanDim, KeepDim(..), Dim(..))
import Torch.Layer.MLP (MLPHypParams(..), MLPParams(..), mlpLayer, ActName(..))
import Torch.NN (Parameterized(..), forward, Parameter, Linear, LinearSpec(..), sample)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (eye', zeros')
import Torch.Optim (GD(..), runStep) -- Import direct de GD
import qualified Torch as T

textFilePath = "Session6/data/sample.txt"
modelPath =  "Session6/data/sample_embedding.params"
wordLstPath = "Session6/data/sample_wordlst.txt"

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int,
  wordDim :: Int 
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

data Model = Model {
    mlp :: MLPParams,
    embeddings :: Embedding
  } deriving (Generic, Parameterized) -- Plus besoin d'instance manuelle avec GD !

--STEP 1
isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "!"]

preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (map (C.filter isAlphaNum) . C.words) textLines
  where
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
    textLines = C.lines (C.map toLower filteredtexts)

-- STEP 2
wordToIndexFactory :: [B.ByteString] -> (B.ByteString -> Int)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))

-- STEP 3
makeTargCont :: [[Int]] -> [([Int], Int)]
makeTargCont linesIdxes = concatMap windowLine linesIdxes
  where
    windowLine line = 
      [ ([line !! (i - 1), line !! (i + 1)], line !! i) 
      | i <- [1 .. length line - 2] ]

-- STEP 4
toyEmbedding :: EmbeddingSpec -> Tensor
toyEmbedding EmbeddingSpec{..} = eye' wordNum wordDim

-- STEP 5
cbow :: Tensor -> Tensor
cbow vec = T.sumDim (Dim 1) KeepDim T.Float vec

-- STEP 6
crossEntropyLoss :: Tensor -> Tensor -> Tensor
crossEntropyLoss predictions target =
  let expScores = T.exp predictions
      sumExp = cbow expScores
      logSumExp = T.log sumExp
      targetScore = T.indexSelect 1 target predictions
  in T.mean (logSumExp - targetScore)

trainStep :: Int -> [([Int], Int)] -> Model -> IO (Model, Float)
trainStep epoch batch model = do
    let contextes = map fst batch 
        cibles    = map (fromIntegral . snd) batch :: [Int64] 

    let xTrain = asTensor contextes 
        yTrain = asTensor cibles    

    let embTrain = embedding' (toDependent $ wordEmbedding $ embeddings model) xTrain
    let bowInput = meanDim (Dim 1) RemoveDim T.Float embTrain

    let y'Train = mlpLayer (mlp model) bowInput
    let trainLoss = crossEntropyLoss y'Train yTrain
        
    let !trainLossValue = asValue trainLoss :: Float

    (newModel, _) <- runStep model GD trainLoss (1e-2 :: Tensor)

    when (epoch `mod` 5 == 0 || epoch == 1) $ do
        putStrLn $ "Epoch " ++ show epoch ++ " | Train Loss: " ++ show trainLossValue
        hFlush stdout

    return (newModel, trainLossValue)

main :: IO ()
main = do
  texts <- B.readFile textFilePath

  let wordLines = preprocess texts
      wordlst = nub $ concat wordLines
      wordToIndex = wordToIndexFactory wordlst
  print wordlst

  let totalWords = length wordlst + 1

  let embsddingSpec = EmbeddingSpec {wordNum = totalWords, wordDim = 9}
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
  let emb = Embedding { wordEmbedding = wordEmb }

  let mlpSpec = MLPHypParams (T.Device T.CPU 0) 9 [(totalWords, Id)]
  mlpComponent <- sample mlpSpec

  let initModel = Model { mlp = mlpComponent, embeddings = emb }

  let idxes = map (map wordToIndex) wordLines
      datasetBatch = filter (\(ctx, _) -> not (null ctx)) (makeTargCont idxes)

  putStrLn "*** Training with GD ***"
  
  (trainedModel, _) <- foldM (\(currentModel, _) epochNum -> do
        (!newModel, !lossVal) <- trainStep epochNum datasetBatch currentModel
        return (newModel, lossVal)
    ) (initModel, 0 :: Float) [1..50]

  putStrLn "*** End Training ***"
  
  saveParams trainedModel modelPath
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  
  return ()