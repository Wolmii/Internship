module PerceptronAndGate where

type Tensor = Double

trainingData :: [([Tensor], Tensor)] --dataset we will be using
trainingData = [([1,1],1),([1,0],0),([1,1],1),([0,0],0)]

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
    let pred = perceptron tanhFunc x w b
        err = calculateError target pred
        newW = zipWith (\acc xi -> acc + lrVal * err * xi) w x
        newB = b + lrVal * err
    in (newW, newB)

-- main, training our model and using it on our datas
main :: IO ()
main = do
    let w = [0.1, 0.2]
    let b = 0.0
    
    let (finalWeights, finalBias) = foldl (\params _ -> 
            foldl (\currentParams example -> trainStep example currentParams lr) params trainingData
          ) (w, b) epoch

    putStrLn "*** Res ***" -- printing our results
    mapM_ (\(x, target) -> do
        let pred = perceptron tanhFunc x finalWeights finalBias
        putStrLn $ "Input: " ++ show x ++ " | Pred: " ++ show pred ++ " (Target: " ++ show target ++ ")"
        ) trainingData
