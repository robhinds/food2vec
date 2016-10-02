package io.github.robhinds.f2v

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.factory.Nd4j

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

object Main extends App {

  val vec = loadExistingFoodModel
  //val vec = trainFoodModelOnText()

  vec.wordsNearest("lamb", 5).foreach{ word =>
    println (s"lamb :: $word = ${vec.similarity("lamb", word)}")
  }
  /**
    * Outputs:
    * lamb :: lean = 0.4001898765563965
    * lamb :: chops = 0.2934269607067108
    * lamb :: shin = 0.4334653615951538
    * lamb :: neck = 0.3357560336589813
    * lamb :: mutton = 0.29669204354286194
    */

  vec.wordsNearest("pork", 5).foreach{ word =>
    println (s"pork :: $word = ${vec.similarity("pork", word)}")
  }
  /**
    * Outputs:
    * pork :: individual = 0.18684068322181702
    * pork :: sourdough = 0.3047674000263214
    * pork :: belly = 0.013941929675638676
    * pork :: unsmoked = 0.26246118545532227
    * pork :: sinew = 0.36444804072380066
    */

  vec.wordsNearest(List("pork", "brisket").asJava, List("beef").asJava, 5).foreach{ word =>
    println(s"Pork equivalent of beef brisket: $word")
  }

  WordVectorSerializer.writeFullModel(vec, "food-model.txt");

  def loadExistingFoodModel() = {
    val sModel = new ClassPathResource("food-model-ingredients-5w300d.txt").getFile()
    WordVectorSerializer.loadFullModel(sModel.getAbsolutePath)
  }

  def trainFoodModelOnText() = {
    val stopWords = List("teaspoon", "tablespoon","tspn", "tblspn", "grams", "litres",
      "pounds", "millilitres", "ounce", "chopped", "salt", "pepper").asJava

    val filePath = new ClassPathResource("sentences.txt").getFile().getAbsolutePath()
    println("Load & Vectorize Sentences....")

    val iter = new BasicLineIterator(filePath)
    val t = new DefaultTokenizerFactory()
    t.setTokenPreProcessor(new CommonPreprocessor())

    println("Building model....")
    Nd4j.factory().setOrder('f')
    val vec = new Word2Vec.Builder()
      .minWordFrequency(10)
      .iterations(10)
      .layerSize(300)
      .seed(42)
      .windowSize(10)
      .iterate(iter)
      .tokenizerFactory(t)
      .stopWords(stopWords)
      .build()
    println("Fitting Word2Vec model....")
    vec.fit()
    vec
  }

}
