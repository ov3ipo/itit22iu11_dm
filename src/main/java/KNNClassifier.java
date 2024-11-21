import weka.core.Instances;
import weka.classifiers.lazy.IBk;

public class KNNClassifier extends BaseClassifier {
    public KNNClassifier() {
        modelName = "KNN";
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        classifier = new IBk(); // Initialize the IBk (KNN) classifier
        classifier.buildClassifier(trainingData); // Train the model
        System.out.println("KNN model built successfully.");
        throw new UnsupportedOperationException("Not implemented yet");
    }
}