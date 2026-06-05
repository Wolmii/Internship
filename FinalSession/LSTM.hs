{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE BangPatterns #-}

module LSTM where

import Codec.Binary.UTF8.String (encode)
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as C
import qualified Data.Map.Strict as M
import qualified Data.ByteString.Internal as B (c2w)
import ML.Exp.Chart (drawLearningCurve)
import GHC.Generics
import Data.Word (Word8)
import Data.Int (Int64)
import Data.List (foldl', nub)
import Data.Char (toLower, isAlphaNum)
import Torch.NN (Parameter, Parameterized(..), Randomizable(..), sample)
import Torch.Serialize (loadParams)
import Torch.TensorFactories (randnIO', eye', zeros')
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', Dim(..))
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Layer.Linear (LinearHypParams(..), LinearParams(..), linearLayer)
import qualified Torch as T
import Control.Monad (foldM, when)
import Torch.Optim (GD(..), runStep)
import System.IO (hFlush, stdout)

data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic, FromJSON, ToJSON)

data AmazonReview = AmazonReview {
  rating :: Float,
  title :: String,
  text :: String,
  images :: [Image],
  asin :: String,
  parent_asin :: String,
  user_id :: String,
  timestamp :: Int,
  verified_purchase :: Bool,
  helpful_vote :: Int
} deriving (Show, Generic, FromJSON, ToJSON)

data ModelSpec = ModelSpec {
  wordNum :: Int, 
  wordDim :: Int  
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
  wordEmbedding :: Parameter
} deriving (Show, Generic, Parameterized)

data LstmSpec = LstmSpec { 
  inputDim  :: Int,
  hiddenDim :: Int
} deriving (Show, Eq, Generic)

data LSTM = LSTM { 
  input_weight  :: Parameter,
  hidden_weight :: Parameter,
  bias          :: Parameter
} deriving (Show, Generic, Parameterized)

data Model = Model {
  emb     :: Embedding,
  lstm     :: LSTM,
  decoder :: LinearParams
} deriving (Show, Generic, Parameterized)

class RecurrentCell cell where
  nextState :: cell -> Tensor -> Tensor -> Tensor

instance RecurrentCell LSTM where
  nextState LSTM {..} input hidden = 
    let ih = toDependent input_weight
        hh = toDependent hidden_weight
        b  = toDependent bias
    in gate input hidden tanhFunc ih hh b

-- the calcul behind the lstm, with weight and bias
gate :: Tensor -> Tensor -> (Tensor -> Tensor) -> Tensor -> Tensor -> Tensor -> Tensor
gate input hidden activ w_ih w_hh b = activ (T.matmul input w_ih + T.matmul hidden w_hh + b)

-- the activation functino we use 
tanhFunc :: Tensor -> Tensor
-- tanhFunc x = (T.exp x - T.exp (-x)) / (T.exp x + T.exp (-x))
tanhFunc x = T.tanh x

-- cut the sentence into words 
unstack :: Tensor -> [Tensor]
unstack t = [T.select 0 i t | i <- [0 .. (head (T.shape t) - 1)]]

instance Randomizable ModelSpec Model where
  sample ModelSpec {..} = 
    Model
    <$> (Embedding <$> (makeIndependent =<< randnIO' [wordNum, wordDim]))
    <*> sample (LstmSpec wordDim 500) 
    <*> sample (LinearHypParams (T.Device T.CPU 0) True 500 5)

instance Randomizable RnnSpec RNN where
  sample RnnSpec {..} = do
    let scale = 1.0 / sqrt (fromIntegral hiddenDim) :: Float
    
    w_ih_raw <- randnIO' [inputDim, hiddenDim]
    w_hh_raw <- randnIO' [hiddenDim, hiddenDim]
    
    let b_raw = zeros' [hiddenDim] 
    
    w_ih <- makeIndependent (w_ih_raw * T.asTensor [scale])
    w_hh <- makeIndependent (w_hh_raw * T.asTensor [scale])
    b    <- makeIndependent b_raw
    
    return $ RNN w_ih w_hh b

-- initialize the model
initialize :: ModelSpec -> {-FilePath -> -}IO Model
initialize modelSpec {-embPath -}= do
  randomizedModel <- sample modelSpec
  {-
  loadedEmb <- loadParams (emb randomizedModel) embPath
  return Model {
    emb = loadedEmb, 
    lstm = lstm randomizedModel, 
    decoder = decoder randomizedModel
  }
  -}
  return randomizedModel

-- let the word go ine by one in the LSTM
forwardRegression :: Model -> Tensor -> [Int64] -> Tensor
forwardRegression model h0 wordIds =
  let xTrain = asTensor wordIds
      wEmb = toDependent (wordEmbedding (emb model))
      embTrain = embedding' wEmb xTrain 
      wordVectors = unstack embTrain
      
      hLast = foldl' (\hBrut x_t -> nextState (lstm model) x_t hBrut) h0 wordVectors
      rawPrediction = linearLayer (decoder model) hLast
      
      batchedPrediction = T.unsqueeze (Dim 0) rawPrediction 
  in batchedPrediction

predictRatingClassification :: Model -> [Int64] -> Int64
predictRatingClassification model wordIds =
  let h0 = T.zeros' [500]
      predTensor = forwardRegression model h0 wordIds
      bestClass = T.argmax (Dim 1) T.RemoveDim predTensor
  in asValue bestClass :: Int64

trainPath :: FilePath
trainPath = "FinalSession/data/train.csv"

testPath :: FilePath
testPath = "FinalSession/data/test.csv"


decodeToAmazonReview :: B.ByteString -> Either String [AmazonReview] 
decodeToAmazonReview jsonl =
  let jsonList = B.split (B.c2w '\n') jsonl
  in sequenceA $ map eitherDecode (filter (not . B.null) jsonList)

-- clear the sentences, with no caps and no symbole
preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (map (C.filter isAlphaNum) . C.words) textLines
  where
    filteredtexts = B.pack $ filter (\w -> w `notElem` map (head . encode) [".", "!"]) (B.unpack texts)
    textLines = C.lines (C.map toLower filteredtexts)

-- get the sentence, and give each word an index 
wordToIndexFactory :: [B.ByteString] -> (B.ByteString -> Int64)
wordToIndexFactory wordlst wrd = 
  --M.findWithDefault (fromIntegral (length wordlst)) wrd (M.fromList (zip wordlst [0..]))
  M.findWithDefault 0 wrd (M.fromList (zip wordlst [0..]))

-- how we transform the floats in star rating 
discretize :: Float -> Float
discretize cosSim
  | cosSim >= 0    && cosSim < 0.5 = 0.0
  | cosSim >= 0.50 && cosSim < 1.5 = 1.0
  | cosSim >= 1.50 && cosSim < 2.5 = 2.0
  | cosSim >= 2.5  && cosSim < 3.5 = 3.0
  | cosSim >= 3.5  && cosSim < 4.5 = 4.0
  | otherwise                      = 5.0

-- the loss
crossEntropyLoss :: Tensor -> Tensor -> Tensor
crossEntropyLoss predictions target =
  let
      maxVal = fst $ T.maxDim (Dim 1) T.KeepDim predictions
      stabilizedPreds = predictions - maxVal
      expScores = T.exp stabilizedPreds
      sumExp = T.sumDim (Dim 1) T.KeepDim T.Float expScores
      logSumExp = T.log sumExp
      targetIdx = fromIntegral (asValue target :: Int64)
      targetScore = T.select 1 targetIdx stabilizedPreds
  in T.mean (logSumExp - targetScore)

trainStep :: [([Int64], Float)] -> Model -> IO (Model, Float)
trainStep batch model = do
  let lr = 0.01
  let hDim = 500

  let totalLoss = foldl' (\accLoss (wordIds, targetRating) ->
          let h0 = T.zeros' [hDim]
              pred = forwardRegression model h0 wordIds
              
              targetIdx = round targetRating - 1 :: Int64
              target = T.asTensor [targetIdx]
              
              loss = crossEntropyLoss pred target
          in accLoss + loss
        ) (T.zeros' [1]) batch

  let meanLoss = totalLoss / T.asTensor [fromIntegral (length batch) :: Float]
  let !lossValue = T.asValue meanLoss :: Float

  (newModel, _) <- runStep model GD meanLoss lr
  return (newModel, lossValue)

epoc :: [Int] --  number of training iteration 
epoc = [1..100]

main :: IO ()
main = do
  wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
  let wordToIndex = wordToIndexFactory wordLst
      totalWords  = length wordLst + 1

  -- training data
  jsonlTrain <- B.readFile trainPath
  let reviewsTrain = case decodeToAmazonReview jsonlTrain of
                       Left err -> []
                       Right r  -> r
      datasetTrain = map (\r -> 
          let tokens = concat $ preprocess (C.pack $ text r)
              ids = map wordToIndex tokens
          in (ids, rating r)
        ) reviewsTrain
      cleanDatasetTrain = filter (\(ids, _) -> not (null ids)) datasetTrain

  -- test data
  jsonlTest <- B.readFile testPath
  let reviewsTest = case decodeToAmazonReview jsonlTest of
                      Left err -> []
                      Right r  -> r
      datasetTest = map (\r -> 
          let tokens = concat $ preprocess (C.pack $ text r)
              ids = map wordToIndex tokens
          in (ids, rating r)
        ) reviewsTest
      cleanDatasetTest = filter (\(ids, _) -> not (null ids)) datasetTest

  let modelSpec = ModelSpec { wordDim = 200, wordNum = 341 }
  initModel <- initialize modelSpec

  putStrLn "*** Training ***"
  putStrLn $ "Entraînement sur : " ++ show (length cleanDatasetTrain) ++ " exemples."
  
  (trainedModel, allLosses) <- foldM (\(currentModel, losses) epochNum -> do
        (newModel, lossVal) <- trainStep cleanDatasetTrain currentModel
        putStrLn $ "Epoch " ++ show epochNum ++ " | Loss : " ++ show lossVal
        hFlush stdout
        return (newModel, losses ++ [lossVal])
    ) (initModel, []) epoc

  let chartData = [("loss", allLosses)]
  drawLearningCurve "lossRNN2.png" "Courbe d'apprentissage RNN" chartData 
  putStrLn "Graph : lossRNN2.png généré."

  putStrLn "*** Eval ***"
  let resultats = map (\(ids, vraieNote) ->
          let indexPredit = predictRatingClassification trainedModel ids
              indexReel   = round vraieNote - 1 :: Int64
          in indexPredit == indexReel
        ) cleanDatasetTest

  let nbCorrects = length (filter id resultats)
      total      = length cleanDatasetTest
      accuracy   = (fromIntegral nbCorrects / fromIntegral total) * 100 :: Float

  putStrLn $ "Correct : " ++ show nbCorrects ++ " / " ++ show total
  putStrLn $ "Accuracy : " ++ show accuracy ++ " %"