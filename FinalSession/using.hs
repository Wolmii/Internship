{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}

module Using where

import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as C
import qualified Data.Map.Strict as M
import GHC.Generics
import Data.Int (Int64)
import Data.List (foldl')
import Data.Char (toLower, isAlphaNum)
import Torch.NN (Parameter, Parameterized(..), Randomizable(..), sample)
import Torch.Serialize (loadParams)
import Torch.TensorFactories (randnIO', zeros')
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', Dim(..))
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Layer.Linear (LinearHypParams(..), LinearParams(..), linearLayer)
import qualified Torch as T
import System.IO (hFlush, stdout)

data ModelSpec = ModelSpec { wordNum :: Int, wordDim :: Int } deriving (Show, Eq, Generic)
data Embedding = Embedding { wordEmbedding :: Parameter } deriving (Show, Generic, Parameterized)
data LstmSpec  = LstmSpec { inputDim :: Int, hiddenDim :: Int } deriving (Show, Eq, Generic)

data LSTM = LSTM { 
  input_weight  :: Parameter,
  hidden_weight :: Parameter,
  bias          :: Parameter
} deriving (Show, Generic, Parameterized)

data Model = Model {
  emb     :: Embedding,
  lstm    :: LSTM,
  decoder :: LinearParams
} deriving (Show, Generic, Parameterized)

class RecurrentCell cell where
  nextState :: cell -> Tensor -> (Tensor, Tensor) -> (Tensor, Tensor)

instance RecurrentCell LSTM where
  nextState LSTM {..} input (hPrev, cPrev) = 
    let ih = T.transpose2D (toDependent input_weight)
        hh = T.transpose2D (toDependent hidden_weight)
        b  = toDependent bias
        gates = T.matmul input ih + T.matmul hPrev hh + b
        hDim   = head (T.shape hPrev)
        chunks = T.split (fromIntegral hDim) (Dim 0) gates
        i_gate = T.sigmoid (chunks !! 0)
        f_gate = T.sigmoid (chunks !! 1)
        g_gate = T.tanh    (chunks !! 2)
        o_gate = T.sigmoid (chunks !! 3)
        cNew = (f_gate * cPrev) + (i_gate * g_gate)
        hNew = o_gate * T.tanh cNew
    in (hNew, cNew)

unstack :: Tensor -> [Tensor]
unstack t = [T.select 0 i t | i <- [0 .. (head (T.shape t) - 1)]]

instance Randomizable ModelSpec Model where
  sample ModelSpec {..} = Model
    <$> (Embedding <$> (makeIndependent =<< randnIO' [wordNum, wordDim]))
    <*> sample (LstmSpec wordDim 128)
    <*> sample (LinearHypParams (T.Device T.CPU 0) True 128 6)

instance Randomizable LstmSpec LSTM where
  sample LstmSpec {..} = do
    let scale = 1.0 / sqrt (fromIntegral hiddenDim) :: Float
    w_ih_raw <- randnIO' [4 * hiddenDim, inputDim]
    w_hh_raw <- randnIO' [4 * hiddenDim, hiddenDim]
    w_ih <- makeIndependent (w_ih_raw * T.asTensor [scale])
    w_hh <- makeIndependent (w_hh_raw * T.asTensor [scale])
    b    <- makeIndependent (zeros' [4 * hiddenDim])
    return $ LSTM w_ih w_hh b

forwardRegression :: Model -> (Tensor, Tensor) -> [Int64] -> Tensor
forwardRegression model states0 wordIds =
  let xTrain = asTensor wordIds
      wEmb = toDependent (wordEmbedding (emb model))
      embTrain = embedding' wEmb xTrain 
      wordVectors = unstack embTrain
      (hLast, _) = foldl' (\states x_t -> nextState (lstm model) x_t states) states0 wordVectors
  in T.unsqueeze (Dim 0) (linearLayer (decoder model) hLast)

predictRatingClassification :: Model -> [Int64] -> Int64
predictRatingClassification model wordIds =
  let predTensor = forwardRegression model (T.zeros' [128], T.zeros' [128]) wordIds
  in asValue (T.argmax (Dim 1) T.RemoveDim predTensor) :: Int64

preprocess :: C.ByteString -> [C.ByteString]
preprocess textBody = map (C.filter isAlphaNum) (C.words (C.map toLower textBody))

sentimentLabel :: Int64 -> String
sentimentLabel l
  | l == 1 || l == 2       = "Nice :)"
  | l == 0 || l == 3 || l == 4 = "Bad :("
  | otherwise              = "Neutral"

modelFile :: FilePath
modelFile = "FinalSession/data/trained_model.params"

vocabFile :: FilePath
vocabFile = "FinalSession/data/vocabulary.txt"

main :: IO ()
main = do
  vocabContent <- B.readFile vocabFile
  let wordLst = C.lines vocabContent
      wordToIndex = \wrd -> M.findWithDefault 0 wrd (M.fromList (zip wordLst [0..]))
      totalWords  = length wordLst + 1

  rawModel <- sample (ModelSpec { wordNum = totalWords, wordDim = 128 })
  
  trainedModel <- loadParams rawModel modelFile
  
  let loop = do
        putStr "\nSay somethings good or bad (exit to quit) : "
        hFlush stdout
        rawInput <- getLine
        if rawInput == "exit"
          then putStrLn "Bye !"
          else do
            let inputLine   = C.pack rawInput
                inputTokens = preprocess inputLine
                inputIds    = map wordToIndex inputTokens
            if null inputIds
              then putStrLn "unkwow sentence"
              else do
                let predictedClass = predictRatingClassification trainedModel inputIds
                putStrLn $ "You are : " ++ sentimentLabel predictedClass
            loop
  loop