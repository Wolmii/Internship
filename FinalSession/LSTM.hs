{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE BangPatterns #-}

module LSTM where

import Codec.Binary.UTF8.String (encode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as C
import qualified Data.Map.Strict as M
import qualified Data.Vector as V
import Data.Csv (FromNamedRecord, decodeByName)
import ML.Exp.Chart (drawLearningCurve)
import GHC.Generics
import Data.Int (Int64)
import Data.List (foldl', nub)
import Data.Char (toLower, isAlphaNum)
import Torch.NN (Parameter, Parameterized(..), Randomizable(..), sample)
import Torch.TensorFactories (randnIO', zeros')
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', Dim(..))
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Layer.Linear (LinearHypParams(..), LinearParams(..), linearLayer)
import qualified Torch as T
import Control.Monad (foldM)
import Torch.Optim (GD(..), runStep)
import System.IO (hFlush, stdout)
import Text.Printf (printf)

import Torch.Serialize (saveParams, loadParams)

data ReviewData = ReviewData {
    text  :: !C.ByteString,
    label :: !Int64
} deriving (Show, Generic, FromNamedRecord)

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
  sample ModelSpec {..} = 
    Model
    <$> (Embedding <$> (makeIndependent =<< randnIO' [wordNum, wordDim]))
    <*> sample (LstmSpec wordDim 500)
    <*> sample (LinearHypParams (T.Device T.CPU 0) True 500 6)

instance Randomizable LstmSpec LSTM where
  sample LstmSpec {..} = do
    let scale = 1.0 / sqrt (fromIntegral hiddenDim) :: Float
    w_ih_raw <- randnIO' [4 * hiddenDim, inputDim]
    w_hh_raw <- randnIO' [4 * hiddenDim, hiddenDim]
    let b_raw = zeros' [4 * hiddenDim] 
    
    w_ih <- makeIndependent (w_ih_raw * T.asTensor [scale])
    w_hh <- makeIndependent (w_hh_raw * T.asTensor [scale])
    b    <- makeIndependent b_raw
    return $ LSTM w_ih w_hh b

initialize :: ModelSpec -> IO Model
initialize modelSpec = sample modelSpec

forwardRegression :: Model -> (Tensor, Tensor) -> [Int64] -> Tensor
forwardRegression model states0 wordIds =
  let xTrain = asTensor wordIds
      wEmb = toDependent (wordEmbedding (emb model))
      embTrain = embedding' wEmb xTrain 
      wordVectors = unstack embTrain
      
      (hLast, _) = foldl' (\states x_t -> nextState (lstm model) x_t states) states0 wordVectors
      rawPrediction = linearLayer (decoder model) hLast
      batchedPrediction = T.unsqueeze (Dim 0) rawPrediction 
  in batchedPrediction

predictRatingClassification :: Model -> [Int64] -> Int64
predictRatingClassification model wordIds =
  let h0 = T.zeros' [500]
      c0 = T.zeros' [500]
      predTensor = forwardRegression model (h0, c0) wordIds
      bestClass = T.argmax (Dim 1) T.RemoveDim predTensor
  in asValue bestClass :: Int64

preprocess :: C.ByteString -> [C.ByteString]
preprocess textBody = map (C.filter isAlphaNum) (C.words (C.map toLower textBody))

wordToIndexFactory :: [C.ByteString] -> (C.ByteString -> Int64)
wordToIndexFactory wordlst wrd = 
  M.findWithDefault 0 wrd (M.fromList (zip wordlst [0..]))

crossEntropyLoss :: Tensor -> Tensor -> Tensor
crossEntropyLoss predictions target =
  let maxVal = fst $ T.maxDim (Dim 1) T.KeepDim predictions
      stabilizedPreds = predictions - maxVal
      expScores = T.exp stabilizedPreds
      sumExp = T.sumDim (Dim 1) T.KeepDim T.Float expScores
      logSumExp = T.log sumExp
      targetIdx = fromIntegral (asValue target :: Int64)
      targetScore = T.select 1 targetIdx stabilizedPreds
  in T.mean (logSumExp - targetScore)

trainStep :: [([Int64], Int64)] -> Model -> IO (Model, Float)
trainStep batch model = do
  let lr = 0.1
  let hDim = 500

  let totalLoss = foldl' (\accLoss (wordIds, targetClass) ->
          let h0 = T.zeros' [hDim]
              c0 = T.zeros' [hDim]
              pred = forwardRegression model (h0, c0) wordIds
              target = T.asTensor [targetClass]
              loss = crossEntropyLoss pred target
          in accLoss + loss
        ) (T.zeros' [1]) batch

  let meanLoss = totalLoss / T.asTensor [fromIntegral (length batch) :: Float]
  let !lossValue = T.asValue meanLoss :: Float

  (newModel, _) <- runStep model GD meanLoss lr
  return (newModel, lossValue)

epoc :: [Int]
epoc = [1..10]

loadDataset :: FilePath -> IO (V.Vector ReviewData)
loadDataset path = do
    csvData <- B.readFile path
    case decodeByName csvData of
        Left err -> error $ "Erreur de lecture du CSV " ++ path ++ " : " ++ err
        Right (_, v) -> return v


embeddingFile :: FilePath
embeddingFile = "FinalSession/data/trained_embedding.params"

sentimentLabel :: Int64 -> String
sentimentLabel l
  | l == 1 || l == 2       = "Nice :)"
  | l == 0 || l == 3 || l == 4 = "Bad :("
  | otherwise              = "Neutral"

confusionMatrix :: [Int64] -> [Int64] -> [Int64] -> [[Int]]
confusionMatrix classes actual predicted =
    let d = zip actual predicted
    in map (\classI -> [length $ filter (\(a, p) -> a == classI && p == classJ) d | classJ <- classes]) classes

printMatrix :: [Int64] -> [[Int]] -> IO ()
printMatrix classes matrix = do
    let sepLine = "+" ++ replicate 10 '-' ++ concatMap (const "+--------") classes ++ "+"

    -- Affichage de l'en-tête du tableau (colonnes des prédictions)
    putStrLn sepLine
    putStr "|          |"
    mapM_ (printf " Pred %-2d|") classes
    putStrLn ""
    putStrLn sepLine

    -- Affichage des lignes de données (valeurs réelles et comptes)
    let printMatrixRow (actualLabel, counts) = do
            printf "| Actual %-2d|" actualLabel
            mapM_ (printf " %6d |") counts
            putStrLn ""
            putStrLn sepLine

    mapM_ printMatrixRow $ zip classes matrix

main :: IO ()
main = do
  trainDataRaw <- loadDataset "FinalSession/data/trainpc.csv"
  testDataRaw  <- loadDataset "FinalSession/data/testpc.csv"

  let allTokens = concatMap (preprocess . text) (V.toList trainDataRaw)
      wordLst   = nub allTokens
      wordToIndex = wordToIndexFactory wordLst
      totalWords  = length wordLst + 1

  let datasetTrain = map (\r -> (map wordToIndex (preprocess (text r)), label r)) (V.toList trainDataRaw)
      cleanDatasetTrain = filter (\(ids, _) -> not (null ids)) datasetTrain

  let datasetTest  = map (\r -> (map wordToIndex (preprocess (text r)), label r)) (V.toList testDataRaw)
      cleanDatasetTest  = filter (\(ids, _) -> not (null ids)) datasetTest

  let modelSpec = ModelSpec { wordNum = totalWords, wordDim = 500 }
  initModel <- initialize modelSpec

  putStrLn "*** Training LSTM on Sentiment CSV ***"
  
  (trainedModel, allLosses) <- foldM (\(currentModel, losses) epochNum -> do
        (newModel, lossVal) <- trainStep cleanDatasetTrain currentModel
        putStrLn $ "Epoch " ++ show epochNum ++ " | Loss : " ++ show lossVal
        hFlush stdout
        return (newModel, losses ++ [lossVal])
    ) (initModel, []) epoc

  saveParams trainedModel "FinalSession/data/trained_model.params"
  C.writeFile "FinalSession/data/vocabulary.txt" (C.unlines wordLst)

  let chartData = [("loss", allLosses)]
  drawLearningCurve "lossLSTM_textCSV.png" "Courbe d'apprentissage LSTM (Text CSV)" chartData 
  putStrLn "Graph : lossLSTM_textCSV.png généré."

  putStrLn "*** Eval ***"

  let targetsRaw = map snd cleanDatasetTest
      predsRaw   = map (\(ids, _) -> predictRatingClassification trainedModel ids) cleanDatasetTest

  let evaluations = map (\(ids, vraiLabel) ->
          let indexPredit = predictRatingClassification trainedModel ids
              correctStrict = indexPredit == vraiLabel
              sentimentPredit = sentimentLabel indexPredit
              sentimentReel   = sentimentLabel vraiLabel
              correctSentiment = sentimentPredit == sentimentReel
          in (correctStrict, correctSentiment)
        ) cleanDatasetTest

  let nbStrictCorrects = length (filter fst evaluations)
      total            = length cleanDatasetTest
      accuracyStricte  = (fromIntegral nbStrictCorrects / fromIntegral total) * 100 :: Float

  let nbSentimentCorrects = length (filter snd evaluations)
      accuracySentiment   = (fromIntegral nbSentimentCorrects / fromIntegral total) * 100 :: Float
  
  let classes = [0, 1, 2, 3, 4, 5]
  let matrix = confusionMatrix classes targetsRaw predsRaw

  putStrLn "\nConfusion Matrix"
  printMatrix classes matrix
  
  putStrLn $ "Good classes   - Correct : " ++ show nbStrictCorrects ++ " / " ++ show total
  putStrLn $ "Good classes   - Accuracy : " ++ show accuracyStricte ++ " %"
  
  putStrLn "----------------------------------------"
  putStrLn $ "Nice/Bad     - Correct : " ++ show nbSentimentCorrects ++ " / " ++ show total
  putStrLn $ "Nice/Bad     - Accuracy : " ++ show accuracySentiment ++ " %"
