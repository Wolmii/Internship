module MultipleRegression where

import Torch.Tensor (Tensor, asTensor, asValue, shape)
import Torch.Functional (matmul, add, transpose2D, sumAll)
import ML.Exp.Chart (drawLearningCurve)

ys :: [Int]
ys = [130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167]
xs :: [Int]
xs = [148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173]
xs2 :: [Int]
xs2 = [111, 136, 170, 179, 263, 236, 111, 120, 260, 100, 260, 222, 160, 152, 234]
alphA :: Tensor
alphA = 0.000035
alphB :: Tensor
alphB = 0.5
epoch :: Int
epoch = 250

linear :: 
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->
    Tensor ->           -- ^ data x: 1 × 10
    Tensor ->
    Tensor                  -- ^ z: 1 × 10
linear (slope, intercept) slope2 input inp = (slope*input)+(slope2*inp)+intercept

cost ::
    Tensor -> -- ^ grand truth: 1 × 10
    Tensor -> -- ^ estimated values: 1 × 10
    Tensor    -- ^ loss: scalar
cost z z' = (1/(2* asTensor (shape z !! 0))) * (sumAll ((z'-z)*(z'-z)))

calculateNewA :: 
     Tensor ->
     Tensor ->
     Tensor
calculateNewA a xEstimated = (asValue a) - (alphA*((1/(asTensor(length xs))) * (sumAll ((asTensor xs)*(xEstimated-(asTensor ys))))))

calculateNewB :: 
     Tensor ->
     Tensor ->
     Tensor
calculateNewB b xEstimated = (asValue b) - (alphB*((1/(asTensor(length xs))) * (sumAll (xEstimated-(asTensor ys)))))

train :: Int -> Tensor -> Tensor -> [Float] -> [Float] -> IO ()
train 0 a a2 b history hist= do
    putStrLn "end"
    let chartData = [("Cost", reverse history)]
    drawLearningCurve "learning_curvemul1.png" "Mon Graphique" chartData
    putStrLn "Graphique généré : learning_curve.png"
    let chartData = [("Cost2", reverse hist)]
    drawLearningCurve "learning_curvemul2.png" "Mon Graphique" chartData
    putStrLn "Graphique généré : learning_curve.png"
train epochs a a2 b history hist = do
    let xEsti = map (\x -> asValue (linear (a, b) (asTensor x)) :: Float) xs xs1
    -- let res = foldl (\acc (x,y) -> acc ++ "correct answer: " ++ show y ++ "\n" ++ "estimated: " ++ show (linear (a, b) (asTensor x)) ++ "\n******\n") ""  (zip xs ys)
    let resCos = cost (asTensor ys) (asTensor xEsti)
    putStr "Cost : "
    print resCos
    let resCos2 = cost (asTensor ys) (asTensor xEsti2)
    putStr "Cost2 : "
    print resCos
    let currentCost = asValue resCos
    let currentCost2 = asValue resCos2
    let newA = calculateNewA a (asTensor xEsti)
    putStr "New A1 : "
    print newA    
    let newA2 = calculateNewA a2 (asTensor xEsti)
    putStr "New A2 : "
    print newA2
    let newB = calculateNewB b (asTensor xEsti)
    putStr "New B : "
    print newB
    train (epochs - 1) newA newA2 newB (currentCost : history) (currentCost2 : hist)

main :: IO ()
main = do
    let sampleA = 0.0
    let sampleB = 0.0
    -- train epoch sampleA sampleB []
    return ()