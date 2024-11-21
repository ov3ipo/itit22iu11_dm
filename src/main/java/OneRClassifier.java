import weka.classifiers.rules.OneR;
import weka.core.Instances;

public class OneRClassifier extends BaseClassifier {

    public OneRClassifier() {
        modelName = "OneR";
    }
    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        classifier = new OneR(); // Initialize the OneR classifier
        ((OneR) classifier).setOptions(new String[] {"-B", "6"}); // Align configuration with Weka
        classifier.buildClassifier(trainingData); // Train the model
        System.out.println("OneR model built successfully.");
    }
}
