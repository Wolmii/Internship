{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

module Admit where

import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V
import GHC.Generics (Generic)
import Torch.Tensor (asTensor, asValue, shape)
import Torch.Functional (matmul, add, transpose2D, sumAll)
import ML.Exp.Chart (drawLearningCurve)

data AdmissionData = AdmissionData
    { serialNo  :: !Int
    , gre       :: !Double
    , toefl     :: !Double
    , rating    :: !Int
    , sop       :: !Double
    , lor       :: !Double
    , cgpa      :: !Double
    , research  :: !Int
    , chance    :: !Double
    } deriving (Generic, Show)
instance FromRecord AdmissionData

type Tensor = Double

lr :: Double --learning rate
lr = 0.1

epoch :: [Int]
epoch = [1..50]

step :: Tensor -> Tensor -- step activation function
step x = if x > 0 then 1 else 0 

sigmoid :: Tensor -> Tensor -- sigmoid activation function
sigmoid x = 1 / (1 + exp (-x))

tanhFunc :: Tensor -> Tensor -- tanh activation function
tanhFunc x = (exp x - exp (-x)) / (exp x + exp (-x))

-- calculation of the perceptron, using an activation function
perceptron :: (Tensor -> Tensor) -> [Tensor] -> [Tensor] -> Tensor -> Tensor
perceptron act x w b = act $ sum (zipWith (*) x w) + b

-- calculate the difference between what we expected and the result
calculateError :: Tensor -> Tensor -> Tensor
calculateError y x = y-x

-- function to train our model, to adjust the weight and the bias
trainStep :: ([Tensor], Tensor) -> ([Tensor], Tensor) -> Double -> ([Tensor], Tensor)
trainStep (x, target) (w, b) lrVal =
    let pred = perceptron sigmoid x w b
        err = calculateError target pred
        newW = zipWith (\acc xi -> acc + lrVal * err * xi) w x
        newB = b + lrVal * err
    in (newW, newB)

normalize :: Double -> Double
normalize x = (x - 260) / (340 - 260)

loadDataset :: FilePath -> IO (V.Vector AdmissionData)
loadDataset path = do
    csvData <- BL.readFile path
    case decode HasHeader csvData of
        Left err -> do
            putStrLn $ "Error parsing " ++ path ++ ": " ++ err
            return V.empty
        Right rows -> return rows

-- EVALUATE
precision :: Float -> Float -> Float -- x=TP, y=FP
precision x y = x/(x+y) 

recall :: Float -> Float -> Float -- x=TP, y=FN
recall x y = x/(x+y) 

accuracy :: Float -> Float -> Float -> Float -- micro-F1 score
accuracy x y z = x/(x+0.5*(y+z)) 

-- confusion matrix
confusionMatrix :: [Int] -> [Int] -> [[Int]]
confusionMatrix real pred = [[tn, fp], [fn, tp]]
  where
    pairs = zip real pred
    count (t, p) = length $ filter (\x -> x == (t, p)) pairs
    tn = count (0, 0) -- TN
    tp = count (1, 1) -- TP
    fp = count (0, 1) -- FP
    fn = count (1, 0) -- FN

-- F1-score for each class, micro-F1 score, macro-F1 score and weighted F1-score

f1 :: Float -> Float -> Float -> Float -- the same as accuracy, different use
f1 x y z = x/(x+0.5*(y+z)) 

-- survey on loss 

--entropy 
entropy :: [Tensor] -> Tensor
entropy x = - sum [y * log y | y <- x, y > 0]

-- cross-Entropy
crossEntropy :: [Tensor] -> [Tensor] -> Tensor
crossEntropy x y = - sum (zipWith (\xi yi -> xi * log (max yi 1e-15)) x y)

-- KL divergence
klDivergence :: [Tensor] -> [Tensor] -> Tensor
klDivergence x y = sum (zipWith (\xi yi -> xi * log (xi / (max yi 1e-15))) x y)

-- main, training our model and using it on our datas
main :: IO ()
main = do
    let w = [0.1]
    let b = 0.0
    trainData <- loadDataset "Session5/data/train.csv"
    let trainGre = V.map gre trainData
    let normGre = V.map normalize trainGre
    let trainChance = V.map chance trainData
    -- validData <- loadDataset "Session3/data/valid.csv"
    -- let validGre = V.map gre validData
    -- let validChance = V.map chance validData
    evalData <- loadDataset "Session5/data/eval.csv"
    let evalGre = V.map gre evalData
    let normevalgre = V.map normalize evalGre
    let evalChance = V.map chance evalData
    let binreal = map (\c -> if c < 0.5 then 0 else 1) (V.toList evalChance) -- binary of the real admittion
    
    let (finalWeights, finalBias) = foldl (\params _ -> 
            foldl (\currentParams example -> trainStep example currentParams lr) params (zip (map (\x -> [x]) (V.toList normGre)) (V.toList trainChance))
          ) (w, b) epoch

    -- List of binary of our predicxtion
    let binpredi = map (\x -> if perceptron sigmoid [x] finalWeights finalBias < 0.5 then 0 else 1) (V.toList normevalgre)

    let matr = confusionMatrix binreal binpredi -- using ou confusion matrix on our binary lists

    putStrLn "*** Res ***" -- printing our results
    mapM_ (\(x, binreal) -> do
        let pred = perceptron sigmoid x finalWeights finalBias
        let binpred = if pred < 0.5 then 0 else 1
        putStrLn $ "Input: " ++ show x ++ " | Pred: " ++ show binpred ++ " (Target: " ++ show binreal ++ ")"
        ) (zip (map (\x -> [x]) (V.toList normevalgre)) binreal)

    let [[tn, fp], [fn, tp]] = matr -- taking the values of the matrix 

    -- printing the matrix's results
    putStrLn $ "TN : " ++ show tn
    putStrLn $ "FP : " ++ show fp
    putStrLn $ "FN : " ++ show fn
    putStrLn $ "TP : " ++ show tp

    -- evaluation : 
    let precis = precision (fromIntegral tp) (fromIntegral fp)
    let reca = recall (fromIntegral tp) (fromIntegral fn) 
    let acc = accuracy (fromIntegral tp) (fromIntegral fp) (fromIntegral fn)  

    putStrLn $ "precicsion  : " ++ show precis
    putStrLn $ "recall  : " ++ show reca
    putStrLn $ "accuracy  : " ++ show acc

    let fone = f1 (fromIntegral tp) (fromIntegral fp) (fromIntegral fn)  

    putStrLn $ "f1  : " ++ show fone

    mapM_ (\(x, target) -> do
        let p = perceptron sigmoid x finalWeights finalBias
        
        let distReal = [target, 1 - target]
        let distPred = [p, 1 - p]

        let ce = lossCrossEntropy distReal distPred

        let kl = lossKLDivergence distReal distPred

        let nll = if target > 0.5 then lossNLL p else lossNLL (1-p)

        putStrLn $ "Target: " ++ show target ++ " | Pred: " ++ show (round (p*100)/100)
        putStrLn $ "  - Cross-Entropy : " ++ show ce
        putStrLn $ "  - KL Divergence : " ++ show kl
        putStrLn $ "  - NLL           : " ++ show nll
        putStrLn "  ----------------"
        ) (zip evalInputs evalTargets)
    
