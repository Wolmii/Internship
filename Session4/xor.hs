{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Xor where

import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch

--------------------------------------------------------------------------------
-- MLP define a new type of datas, structure of the model
--------------------------------------------------------------------------------

data MLPSpec = MLPSpec
  { feature_counts :: [Int],  -- number of elements of each layers (and number of layers)
    nonlinearitySpec :: Tensor -> Tensor -- function used between layers
  } --structure

data MLP = MLP
  { layers :: [Linear], -- more specific, b and w
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)

instance Randomizable MLPSpec MLP where -- turn mlpspec into mlp
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts -- generate the number of input that we need fot the layers
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes -- initialize them
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec} -- package them for the mlp form
    where -- how to do them ?
      mkLayerSizes (a : (b : t)) =
        scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

mlp :: MLP -> Tensor -> Tensor -- pass the function at every layers
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize = 2

numIters = 2000

model :: MLP -> Tensor -> Tensor
model params t = mlp params t

 -- ^ initialisation, call to mlp as the model 

main :: IO ()
main = do
  init <-
    sample $
      MLPSpec
        { feature_counts = [2, 2, 1],
          nonlinearitySpec = Torch.tanh
        } -- put in the new type and initialize
  trained <- foldLoop init numIters $ \state i -> do --train for the number of itaration
    input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5) -- generate the values
    let (y, y') = (tensorXOR input, squeezeAll $ model state input) -- the "real result", and the result we have
        loss = mseLoss y y' -- the error
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss -- print every 100 steps
    (newState, _) <- runStep state optimizer loss 1e-1
    return newState
  putStrLn "Final Model:" -- after the train, try manually with the answer that we know, to see if it works
  putStrLn $ "0, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 0 :: Float]))
  putStrLn $ "0, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 1 :: Float]))
  putStrLn $ "1, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 0 :: Float]))
  putStrLn $ "1, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 1 :: Float]))
  return ()
  where 
    optimizer = GD -- define the xor that we kn ow, and that the model is trying to recreate
    tensorXOR :: Tensor -> Tensor -- the xor core, when it define if 0 or 1
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = select 1 0 t
        b = select 1 1 t