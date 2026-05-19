{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Bow where
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import Data.Word (Word8)
import qualified Data.Map.Strict as M -- add containers to dependencies in package.yaml
import Data.List (nub)
import Data.Char (toLower)
import Data.Char (isAlphaNum)

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding')
import Torch.NN (Parameterized(..), Parameter)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor)
import Torch.TensorFactories (eye', zeros')

-- your text data (try small data first)
textFilePath = "data/sample.txt"
modelPath =  "data/sample_embedding.params"
wordLstPath = "data/sample_wordlst.txt"

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

-- Probably you should include model and Embedding in the same data class.
data Model = Model {
		mlp :: MLP -- réseau de neuronnes
    embeddings :: Embedding -- table des numéros données par les mots
  } deriving (Show, Generic, Parameterized)

data MLP = MLP {
    mlpWeights :: [Tensor], -- Liste de taille 'wordNum', où chaque Tensor est de taille 9
    mlpBias    :: [Tensor]  -- Liste de biais de taille 'wordNum'
} deriving (Show)

-- STEP 1
isUnncessaryChar :: -- get rid of useless points
  Word8 ->
  Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "!", "&", "$", "€", "=", "+", "(", ")", "{", "}", "/", "[", "]"]

preprocess :: -- preprocess the text, separating line  by line and word by word, get rid of uppercase and symbols
  B.ByteString -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (filter isAlphaNum (toLower (B.split (head $ encode " ")))) textLines -- word by word
  where
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
    textLines = B.split (head $ encode "\n") filteredtexts -- line by line

-- STEP 2
wordToIndexFactory :: -- take the words and create an index based on them
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))

-- STEP 3 separating the data in target and context
makeTargCont :: [[Int]] -> [([Int], Int)]
makeTargCont linesIdxes = concatMap windowLine linesIdxes
  where
    windowLine line = 
      [ ([line !! (i - 1), line !! (i + 1)], line !! i) 
      | i <- [1 .. length line - 2] ]

-- STEP 4
toyEmbedding ::
  EmbeddingSpec ->
  Tensor           -- embedding
toyEmbedding EmbeddingSpec{..} = 
  eye' wordNum wordDim

-- STEP 5 BOW
cbow :: Tensor -> Int -> Tensor
cbow vec size = meanDim size (0) vec

-- STEP 6 MLP 
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
trainStep :: ([Tensor], Tensor) -> ([Tensor], Tensor) -> Double -> Embedding -> ([Tensor], Tensor, Embedding)
trainStep (x, target) (w, b) lrVal emb =
    let pred = perceptron tanhFunc x w b
        err = calculateError target pred
        newW = zipWith (\acc xi -> acc + lrVal * err * xi) w x
        newB = b + lrVal * err
        (newEmb, _) <- runStep model optimizer err 1.0
    in (newW, newB, newEmb)

-- La fonction prend le modèle complet, l'optimiseur, et le couple (Contexte, Cible)
-- Elle renvoie le modèle mis à jour et la valeur de l'erreur pour affichage
trainStep :: Model -> Optimizer -> ([Int], Int) -> IO (Model, Tensor)
trainStep model optimizer (contextIndices, targetIndex) = do
    
    -- 1. Récupérer les embeddings du contexte et faire la fusion BoW (Moyenne)
    let contextTensor = asTensor contextIndices
        embVectors    = embedding' (toDependent $ wordEmbedding (embeddings model)) contextTensor
        bowInput      = meanDim (0) embVectors -- Fusion de tes deux vecteurs en un seul
    
    -- 2. Passer le vecteur fusionné dans le MLP pour obtenir les scores
    let predictions = mlpForward (mlp model) bowInput
        targetTensor = asTensor targetIndex
    
    -- 3. Calculer l'erreur globale (Loss) avec la Cross Entropy de Torch
    let loss = crossEntropyLoss predictions targetTensor
    
    -- 4. Demander à LibTorch de mettre à jour TOUT le modèle d'un coup (MLP + Embedding)
    (updatedModel, _) <- runStep model optimizer loss 1.0
    
    return (updatedModel, loss)

main :: IO ()
main = do
  -- load text file
  texts <- B.readFile textFilePath

  -- STEP 1
  -- Create a unique word list
  let wordLines = preprocess texts
      wordlst = nub $ concat wordLines
      wordToIndex = wordToIndexFactory wordlst
  print wordlst

  -- Create initial embedding (wordDim × wordNum)
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst + 1, wordDim = 9}
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
  let emb = Embedding { wordEmbedding = wordEmb }

  let sampleTxt = B.pack $ encode "This is awesome.\nmodel is developing"
  -- convert word to index STEP 2
      idxes = map (map wordToIndex) (preprocess sampleTxt)
  -- convert to embedding STEP 4
      -- embTxt = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor idxes)
  -- STEP 3 
  let dataset = makeCbowData idxes
  let (firstContext, firstTarget) = head dataset
  -- STEP 4
  let embContext = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor firstContext)
      
　-- TODO: Train model. After training, we can obtain the trained patameter, embeddings. This is the trained embedding.

  -- STEP 5
  let fusion = bow embContext wordDim

-- have to : 
-- 1 separate target and context
-- 2 BOW
-- 3 MLP with random weight, and biais
-- 4 maj weight and embedded vectors (again, and again)
-- 5 save

  -- STEP 6
  let w = [0.1, 0.2]
  let b = 0.0

  putStrLn "*** Training ***"

  let (finalWeights, finalBias, trainedEmb) = foldl (\params _ -> 
    foldl (\currentParams example -> trainStep example currentParams lr) params fusion
    ) (w, b) epoch  

  let (finalWeights, finalBias) = foldl (\params _ -> 
            foldl (\currentParams example -> trainStep example currentParams lr) params (zip (fusion) (firstTarget))
          ) (w, b) epoch

  putStrLn "*** End Training ***"

  -- STEP 7
  -- Save params to use trained parameter in the next session
  -- trainedEmb :: Embedding
  saveParams trainedEmb modelPath
  putStrLn trainedEmb
  -- Save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  
  -- Load params
  -- initWordEmb <- makeIndependent $ zeros' [1]
  -- let initEmb = Embedding {wordEmbedding = initWordEmb}
  -- loadedEmb <- loadParams initEmb modelPath

  return ()