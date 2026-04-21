import Torch.Tensor (Tensor, asTensor, asValue, shape)
import Torch.Functional (matmul, add, transpose2D, sumAll)
import ML.Exp.Chart (drawLearningCurve)

ys :: [Int]
ys = [130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167]
xs :: [Int]
xs = [148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173]
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
     Tensor
calculateNewA a xEstimated = (asValue a) - (alphA*((1/(asTensor(length xs))) * (sumAll ((asTensor xs)*(xEstimated-(asTensor ys))))))

calculateNewB :: 
     Tensor ->
     Tensor ->
     Tensor
calculateNewB b xEstimated = (asValue b) - (alphB*((1/(asTensor(length xs))) * (sumAll (xEstimated-(asTensor ys)))))

train :: Int -> Tensor -> Tensor -> [Float] -> IO ()
train 0 a b history = do
    putStrLn "end"
    let chartData = [("Cost", reverse history)]
    drawLearningCurve "learning_curve.png" "Mon Graphique" chartData
    putStrLn "Graphique généré : learning_curve.png"
train epochs a b history = do
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

main :: IO ()
main = do
    let sampleA = 0.0
    let sampleB = 0.0
    train epoch sampleA sampleB []
    return ()