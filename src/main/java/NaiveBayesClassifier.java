import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class NaiveBayesClassifier extends BaseClassifier {

    public NaiveBayesClassifier() {
        modelName = "NaiveBayes";
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        classifier = new NaiveBayes(); // Initialize the NaiveBayes classifier
        classifier.buildClassifier(trainingData); // Train the model
        System.out.println("NaiveBayes model built successfully.");
    }

}
