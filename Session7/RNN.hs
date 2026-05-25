{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE BangPatterns #-}

module Rnn where


import Codec.Binary.UTF8.String (encode)
import GHC.Generics
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as C
import Data.Word (Word8)
import qualified Data.Map.Strict as M
import Data.List (nub, foldl')
import Data.Char (toLower, isAlphaNum)
import Data.Int (Int64)
import Control.Monad (when, foldM)
import System.IO (hFlush, stdout)

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', meanDim, KeepDim(..), Dim(..))
import Torch.Layer.MLP (MLPHypParams(..), MLPParams(..), mlpLayer, ActName(..))
import Torch.NN (Parameterized(..), forward, Parameter, Linear, LinearSpec(..), sample)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (eye', zeros')
import Torch.Optim (GD(..), runStep)
import qualified Torch as T
import qualified Data.ByteString.Lazy.Char8 as C
import ML.Exp.Chart (drawLearningCurve)
import Data.List.Split (splitOn)

data ElmanSpec = ElmanSpec {in_features :: Int, hidden_features :: Int} -- numbers of features

data ElmanCell = ElmanCell -- data form, datas for the maths part
  { input_weight :: Parameter,
    hidden_weight :: Parameter,
    bias :: Parameter
  } deriving (Show, Parameterized)

instance Randomizable ElmanSpec ElmanCell where -- initialize a randoms cell (as elemSpec chose the  numbers), to start training 
  sample ElmanSpec {..} = do
    w_ih <- makeIndependent =<< randnIO' [in_features, hidden_features]
    w_hh <- makeIndependent =<< randnIO' [hidden_features, hidden_features]
    b <- makeIndependent =<< randnIO' [1, hidden_features]
    return $ ElmanCell w_ih w_hh b