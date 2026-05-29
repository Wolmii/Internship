{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module RnnHask where

import Control.Monad.State.Strict
import Data.List (foldl', intersperse, scanl')
import RecurrentLayer
import Torch

data ElmanSpec = ElmanSpec {in_features :: Int, hidden_features :: Int} -- numbers of features

data ElmanCell = ElmanCell -- data form, datas for the maths part
  { input_weight :: Parameter,
    hidden_weight :: Parameter,
    bias :: Parameter
  }

instance RecurrentCell ElmanCell where -- the gate : do the maths to choses what change (what is keeped, what is updated)
  nextState ElmanCell {..} input hidden =
    gate input hidden Torch.tanh input_weight hidden_weight bias

instance Randomizable ElmanSpec ElmanCell where -- initialize a randoms cell (as elemSpec chose the  numbers), to start training 
  sample ElmanSpec {..} = do
    w_ih <- makeIndependent =<< randnIO' [in_features, hidden_features]
    w_hh <- makeIndependent =<< randnIO' [hidden_features, hidden_features]
    b <- makeIndependent =<< randnIO' [1, hidden_features]
    return $ ElmanCell w_ih w_hh b

instance Parameterized ElmanCell where -- change the parameters, updates them with the enxts ones
  flattenParameters ElmanCell {..} = [input_weight, hidden_weight, bias]
  _replaceParameters _ = do
    input_weight <- nextParameter
    hidden_weight <- nextParameter
    bias <- nextParameter
    return $ ElmanCell {..}

instance Show ElmanCell where -- a print
  show ElmanCell {..} =
    (show input_weight) ++ "\n"
      ++ (show hidden_weight)
      ++ "\n"
      ++ (show bias)
      ++ "\n"