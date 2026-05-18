module Evaluation where

-- Accuracy, precision and recall

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

macro :: Float -> Float -> Float -> Float
macro x y z = (x+y+z)/3

weighted :: Float -> Float -> Float -> Float -> Float -> Float -> Float
weighted x y z xp yp zp = (x*xp)+(y*zp)+(z*zp)

