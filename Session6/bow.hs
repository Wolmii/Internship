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
import Torch.Optim (GD(..), runStep)
import qualified Torch as T
import qualified Data.ByteString.Lazy.Char8 as C
import ML.Exp.Chart (drawLearningCurve)
import Data.List.Split (splitOn)

textFilePath = "Session6/data/sample.txt"
modelPath =  "Session6/data/sample_embedding.params"
wordLstPath = "Session6/data/sample_wordlst.txt"
newPath = "Session6/data/data.txt"

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
  } deriving (Generic, Parameterized)

data File3 = File3 { -- datatype for the format of the new data txt 
    score :: Float,
    sent01 :: C.ByteString,
    sent02 :: C.ByteString
} deriving (Show)

--STEP 1
isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "!"]

preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (map (C.filter isAlphaNum) . C.words) textLines
  where
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
    textLines = C.lines (C.map toLower filteredtexts)

parseLineBasique :: C.ByteString -> [File3]
parseLineBasique line = 
  let colonnes = C.split '\t' line
    in if length colonnes == 3 && not (C.null (colonnes !! 0))
        then
            -- good line 
            let labelRaw = colonnes !! 0
                sent1    = colonnes !! 1
                sent2    = colonnes !! 2
                score    = read (C.unpack labelRaw) :: Float
            in [File3 score sent1 sent2]
        else 
            -- if more or not enought collumns
            []

newpreprocess :: FilePath -> IO [File3]
newpreprocess fp = do 
  content <- C.readFile fp
  let allLines = C.lines (C.map toLower content)
  let valid = concatMap parseLineBasique allLines
  return valid 


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
cbow vec = T.meanDim (Dim 0) T.RemoveDim T.Float vec

-- STEP 6
crossEntropyLoss :: Tensor -> Tensor -> Tensor
crossEntropyLoss predictions target =
  let expScores = T.exp predictions
      sumExp = T.sumDim (Dim 1) T.KeepDim T.Float expScores
      logSumExp = T.log sumExp
      targetScore = T.indexSelect 1 target predictions
  in T.mean (logSumExp - targetScore)

lr :: Tensor --learning rate
lr = 0.99

trainStep :: Int -> [([Int], Int)] -> Model -> IO (Model, Float)
trainStep epoch batch model = do
    let contextes = map fst batch 
        cibles    = map (fromIntegral . snd) batch :: [Int64] 

    let xTrain = asTensor contextes 
        yTrain = asTensor cibles    

    let embTrain = embedding' (toDependent $ wordEmbedding $ embeddings model) xTrain
    let bowInput = cbow embTrain

    let y'Train = mlpLayer (mlp model) bowInput
    let trainLoss = crossEntropyLoss y'Train yTrain
        
    let !trainLossValue = asValue trainLoss :: Float

    (newModel, _) <- runStep model GD trainLoss (lr)

    when (epoch `mod` 5 == 0 || epoch == 1) $ do
        -- putStrLn $ "Epoch " ++ show epoch ++ " | Train Loss: " ++ show trainLossValue
        hFlush stdout

    return (newModel, trainLossValue)

searchWord :: String -> IO Tensor
searchWord wordStr = do
  let word = C.pack wordStr
  txt <- B.readFile wordLstPath
  let wordlst = C.lines txt
      wordToIndex = wordToIndexFactory wordlst
      totalWords = length wordlst + 1
      index = wordToIndex word -- get the index of the word 
  
  if index == totalWords 
    then do 
      putStrLn "Not in file"
      return (T.zeros' [9])
    else do 
      -- create the new embedding 
      let embsddingSpec = EmbeddingSpec {wordNum = totalWords, wordDim = 9}
      wordEmb <- makeIndependent $ T.zeros' [totalWords, 9]
      
      let mlpSpec = MLPHypParams (T.Device T.CPU 0) 9 [(totalWords, Id)]
      mlpComponent <- sample mlpSpec
      let emptyModel = Model { mlp = mlpComponent, embeddings = Embedding { wordEmbedding = wordEmb } }

      -- loads the params after training
      newE <- loadParams emptyModel modelPath

      -- extract the line corresponding to the index 
      let embMatrix = toDependent $ wordEmbedding $ embeddings newE
          vec = T.indexSelect 0 (asTensor [fromIntegral index :: Int64]) embMatrix
      return vec

trainCompare :: File3 -> IO Float
trainCompare fils3 = do 
  let sent1pre = concat $ preprocess (sent01 fils3)-- preprocess on the valid sentences
      sent2pre = concat $ preprocess (sent02 fils3)
  
  vecsent1 <- mapM (\w -> searchWord (C.unpack w)) sent1pre-- search all the words in the embedded trained 
  vecsent2 <- mapM (\w -> searchWord (C.unpack w)) sent2pre
  
  let mat1 = T.stack (Dim 0) vecsent1
      mat2 = T.stack (Dim 0) vecsent2

  let vec1 = cbow mat1-- create a vector with this
      vec2 = cbow mat2
  
  let dotProduct = T.sumAll (vec1 * vec2)-- compare the two
  let norm1 = T.sqrt (T.sumAll (vec1 * vec1))
      norm2 = T.sqrt (T.sumAll (vec2 * vec2))
  let scoreSim = dotProduct / (norm1 * norm2)

  putStrLn $ "Ours : " ++ show scoreSim ++ " Real : " ++ show (score fils3)-- compare to the score
  return (asValue scoreSim :: Float)
  
discretize :: Float -> Float
discretize cosSim
  | cosSim >= (-1.0) && cosSim < (-0.6) = 0.0
  | cosSim >= (-0.6) && cosSim < (-0.2) = 1.0
  | cosSim >= (-0.2) && cosSim < 0.2    = 2.0
  | cosSim >= 0.2    && cosSim < 0.5    = 3.0
  | cosSim >= 0.5    && cosSim < 0.8    = 4.0
  | otherwise                           = 5.0

epoc :: [Int]
epoc = [1..1000]

main :: IO ()
main = do
  pairesFiltrees <- newpreprocess newPath
  cosinusList <- mapM trainCompare pairesFiltrees
  
  let predictions = map discretize cosinusList
      vraisScores = map score pairesFiltrees 
      
  let arrondisEgaux = zipWith (\pred reel -> pred == fromIntegral (round reel)) predictions vraisScores
      nbCorrects    = length (filter id arrondisEgaux)
      total         = length pairesFiltrees
      accuracy      = (fromIntegral nbCorrects / fromIntegral total) * 100 :: Float

  putStrLn $ "NB same : " ++ show nbCorrects ++ " / " ++ show total
  putStrLn $ "Accuracy : " ++ show accuracy ++ " %"
  
  return ()
{-
  texts <- B.readFile textFilePath

  let wordLines = preprocess texts
      wordlst = nub $ concat wordLines
      wordToIndex = wordToIndexFactory wordlst
  --print wordlst

  let totalWords = length wordlst + 1

  let embsddingSpec = EmbeddingSpec {wordNum = totalWords, wordDim = 9}
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
  let emb = Embedding { wordEmbedding = wordEmb }

  let mlpSpec = MLPHypParams (T.Device T.CPU 0) 9 [(totalWords, Id)]
  mlpComponent <- sample mlpSpec

  let initModel = Model { mlp = mlpComponent, embeddings = emb }

  let idxes = map (map wordToIndex) wordLines
      datasetBatch = filter (\(ctx, _) -> not (null ctx)) (makeTargCont idxes)

  putStrLn "*** Training ***"
  
  (trainedModel, allLosses) <- foldM (\(currentModel, losses) epochNum -> do
        (!newModel, !lossVal) <- trainStep epochNum datasetBatch currentModel
        when (epochNum `mod` 5 == 0 || epochNum == 1) $ do
          putStrLn $ "Epoch " ++ show epochNum ++ " | Train Loss: " ++ show lossVal
          hFlush stdout
        return (newModel, losses ++ [lossVal])
    ) (initModel, []) epoc
  putStrLn "*** End Training ***"
  
  saveParams trainedModel modelPath
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)

  vecLove <- searchWord "love"
  print vecLove
  let chartData = [("loss", allLosses)]
  drawLearningCurve "loss.png" "Mon Graphique" chartData 
  putStrLn "Graph : loss.png"

  -- Load params
  -- initWordEmb <- makeIndependent $ zeros' [1]
  -- let initEmb = Embedding {wordEmbedding = initWordEmb}
  -- loadedEmb <- loadParams initEmb modelPath
  return ()
-}