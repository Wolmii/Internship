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
alphA = 0.000001
alphB :: Tensor
alphB = 0.00005
epoch :: Int
epoch = 40

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
calculateNewA a xEstimated xs ys = (asValue a) - (alphA*((1/(asTensor(length (asValue xs :: [Float])))) * (sumAll ((asTensor xs)*(xEstimated-(asTensor ys))))))

calculateNewB :: 
     Tensor ->
     Tensor ->
     Tensor ->
     Tensor ->
     Tensor
calculateNewB b xEstimated xs ys = (asValue b) - (alphB*((1/(asTensor(length (asValue xs :: [Float])))) * (sumAll (xEstimated-(asTensor ys)))))

train :: Int -> Tensor -> Tensor -> [Float] -> Tensor -> Tensor -> Tensor -> Tensor -> [Float] -> IO (Tensor, Tensor)
train 0 a b history xs ys validx validy costValid = do
    putStrLn "end"
    let chartData = [("Cost", reverse history)]
    drawLearningCurve "learning_curve.png" "Mon Graphique" chartData
    putStrLn "Graphique généré : learning_curve.png"
    let chartData2 = [("Cost", reverse costValid)]
    drawLearningCurve "learningValid_curve.png" "Mon Graphique" chartData2
    putStrLn "Graphique généré : learningValid_curve.png"
    return (a, b)
train epochs a b history xs ys validx validy costValid = do
    let xEsti = map (\x -> asValue (linear (a, b) (asTensor x)) :: Float) (asValue xs :: [Float])
    let res = foldl (\acc (x,y) -> acc ++ "correct answer: " ++ show y ++ "\n" ++ "estimated: " ++ show (linear (a, b) (asTensor x)) ++ "\n******\n") ""  (zip (asValue xs :: [Float]) (asValue ys :: [Float]))
    let resCos = cost (asTensor ys) (asTensor xEsti)
    putStr "Cost : "
    print resCos
    let currentCost = asValue resCos
    let newA = calculateNewA a (asTensor xEsti) xs ys
    putStr "New A : "
    print newA    
    let newB = calculateNewB b (asTensor xEsti) xs ys
    putStr "New B : "
    print newB
    let validation = (asValue (valid newA newB validx validy) :: Float) 
    train (epochs - 1) newA newB (currentCost : history) xs ys validx validy (validation : costValid)

instance FromRecord AdmissionData

valid :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
valid newA newB xs ys = 
    let res1  = linear (newA, newB) xs
    in cost ys res1
    -- putStr "Cost final : "
    -- print final

loadDataset :: FilePath -> IO (V.Vector AdmissionData)
loadDataset path = do
    csvData <- BL.readFile path
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
    let sampleA = 0.0
    let sampleB = 0.0
    (a, b) <- train epoch sampleA sampleB [] (asTensor (V.toList trainGre)) (asTensor (V.toList trainChance)) (asTensor (V.toList validGre)) (asTensor (V.toList validChance)) []
    let validation = valid a b (asTensor (V.toList evalGre)) (asTensor (V.toList evalChance))
    let predict = linear (a,b) (asTensor (V.toList validGre))
    let end = foldl (\acc (x,y) -> acc ++ "correct answer: " ++ show y ++ "\n" ++ "estimated: " ++ show x ++ "\n******\n") ""  (zip (asValue predict :: [Float]) (asValue (asTensor (V.toList validChance)) :: [Float]))
    putStrLn end
    let costEnd = cost (asTensor (V.toList validChance)) predict
    let chartData = [("prediction", (asValue costEnd :: [Float]))]
    drawLearningCurve "learning_curveEnd.png" "Mon Graphique" chartData 
    putStrLn "Graphique généré : learning_curveEnd.png"