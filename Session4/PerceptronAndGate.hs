module PerceptronAndGate where

type Tensor = Double

trainingData :: [([Tensor], Tensor)]
trainingData = [([1,1],1),([1,0],0),([1,1],1),([0,0],0)]

lr :: Double
lr = 0.1

step :: Tensor -> Tensor
step x = if x > 0 then 1 else 0 

sigmoid :: Tensor -> Tensor
sigmoid x = 1 / (1 + exp (-x))

tanhFunc :: Tensor -> Tensor
tanhFunc x = (exp x - exp (-x)) / (exp x + exp (-x))

perceptron :: (Tensor -> Tensor) -> [Tensor] -> [Tensor] -> Tensor -> Tensor
perceptron act x w b = act $ sum (zipWith (*) x w) + b

calculateError :: Tensor -> Tensor -> Tensor
calculateError y x = y-x

trainStep :: ([Tensor], Tensor) -> ([Tensor], Tensor) -> Double -> ([Tensor], Tensor)
trainStep (x, target) (w, b) lrVal =
    let pred = perceptron tanhFunc x w b
        err = calculateError target pred
        newW = zipWith (\acc xi -> acc + lrVal * err * xi) w x
        newB = b + lrVal * err
    in (newW, newB)

main :: IO ()
main = do
    let w = [0.1, 0.2]
    let b = 0.0
    let epoch = [1..50]
    
    let (finalWeights, finalBias) = foldl (\params _ -> 
            foldl (\currentParams example -> trainStep example currentParams lr) params trainingData
          ) (w, b) epoch

    putStrLn "*** Res ***"
    mapM_ (\(x, target) -> do
        let pred = perceptron tanhFunc x finalWeights finalBias
        putStrLn $ "Input: " ++ show x ++ " | Pred: " ++ show pred ++ " (Target: " ++ show target ++ ")"
        ) trainingData
