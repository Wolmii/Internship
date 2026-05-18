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
		mlp :: MLP
    embeddings :: Embedding
  } deriving (Show, Generic, Parameterized)

data Use = Use {
    embed :: Embedding
    mod :: Model
} deriving (Show, Generic, Parameterized)

isUnncessaryChar :: -- get ride of useless points
  Word8 ->
  Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "!"]

preprocess :: -- preprocess the text, separating line  by line and word by word 
  B.ByteString -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines -- word by word
  where
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
    textLines = B.split (head $ encode "\n") filteredtexts -- line by line

wordToIndexFactory :: -- take the words and create an index based on them
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))

toyEmbedding :: -- jsp
  EmbeddingSpec ->
  Tensor           -- embedding
toyEmbedding EmbeddingSpec{..} = 
  eye' wordNum wordDim


main :: IO ()
main = do
  -- load text file
  texts <- B.readFile textFilePath

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
  -- convert word to index
      idxes = map (map wordToIndex) (preprocess sampleTxt)
  -- convert to embedding
      embTxt = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor idxes)

　-- TODO: Train model. After training, we can obtain the trained patameter, embeddings. This is the trained embedding.
  
  -- Save params to use trained parameter in the next session
  -- trainedEmb :: Embedding
  saveParams trainedEmb modelPath
  -- Save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  
  -- Load params
  -- initWordEmb <- makeIndependent $ zeros' [1]
  -- let initEmb = Embedding {wordEmbedding = initWordEmb}
  -- loadedEmb <- loadParams initEmb modelPath

  return ()