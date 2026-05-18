{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

module Main where

import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V
import GHC.Generics (Generic)
import Torch.Tensor (Tensor, asTensor, asValue, shape)
import Torch.Functional (matmul, add, transpose2D, sumAll)
import ML.Exp.Chart (drawLearningCurve)

data AdmissionData = AdmissionData
    { serialNo  :: !Int
    , gre       :: !Float
    , toefl     :: !Float
    , rating    :: !Int
    , sop       :: !Float
    , lor       :: !Float
    , cgpa      :: !Float
    , research  :: !Int
    , chance    :: !Float
    } deriving (Generic, Show)

alphA :: Tensor
alphA = 0.000035
alphB :: Tensor
alphB = 0.5
epoch :: Int
epoch = 250

linear :: 
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = slope*input + intercept

cost ::
    Tensor -> -- ^ grand truth: 1 × 10
    Tensor -> -- ^ estimated values: 1 × 10
    Tensor    -- ^ loss: scalar
cost z z' = (1/(2* asTensor (shape z !! 0))) * (sumAll ((z'-z)*(z'-z)))

calculateNewA :: 
     Tensor ->
     Tensor ->
     Tensor -> 
     Tensor ->
     Tensor
calculateNewA a xEstimated xs ys = (asValue a) - (alphA*((1/(asTensor(length xs))) * (sumAll ((asTensor xs)*(xEstimated-(asTensor ys))))))

calculateNewB :: 
     Tensor ->
     Tensor ->
     Tensor ->
     Tensor ->
     Tensor
calculateNewB b xEstimated xs ys = (asValue b) - (alphB*((1/(asTensor(length xs))) * (sumAll (xEstimated-(asTensor ys)))))

train :: Int -> Tensor -> Tensor -> [Float] -> Tensor -> Tensor -> IO ()
train 0 a b history xs ys = do
    putStrLn "end"
    let chartData = [("Cost", reverse history)]
    drawLearningCurve "learning_curve.png" "Mon Graphique" chartData
    putStrLn "Graphique généré : learning_curve.png"
train epochs a b history xs ys= do
    let xEsti = map (\x -> asValue (linear (a, b) (asTensor x)) :: Float) xs
    let res = foldl (\acc (x,y) -> acc ++ "correct answer: " ++ show y ++ "\n" ++ "estimated: " ++ show (linear (a, b) (asTensor x)) ++ "\n******\n") ""  (zip xs ys)
    let resCos = cost (asTensor ys) (asTensor xEsti)
    putStr "Cost : "
    print resCos
    let currentCost = asValue resCos
    let newA = calculateNewA a (asTensor xEsti)
    putStr "New A : "
    print newA    
    let newB = calculateNewB b (asTensor xEsti)
    putStr "New B : "
    print newB
    train (epochs - 1) newA newB (currentCost : history)
-- This instance tells Cassava: "Use the field names to parse the CSV"
instance FromRecord AdmissionData

loadDataset :: FilePath -> IO (V.Vector AdmissionData)
loadDataset path = do
    csvData <- BL.readFile path
    -- HasHeader skips the first line (titles). Use NoHeader if there is no title line.
    case decode HasHeader csvData of
        Left err -> do
            putStrLn $ "Error parsing " ++ path ++ ": " ++ err
            return V.empty
        Right rows -> return rows

main :: IO ()
main = do
    trainData <- loadDataset "Session3/data/train.csv"
    let trainGre = V.map gre trainData
    let trainChance = V.map chance trainData
    validData <- loadDataset "Session3/data/valid.csv"
    let validGre = V.map gre validData
    let validChance = V.map chance validData
    evalData <- loadDataset "Session3/data/eval.csv"
    let evalGre = V.map gre evalData
    let evalChance = V.map chance evalData
    train epoch trainGre trainChance []