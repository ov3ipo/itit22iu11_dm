import weka.core.Instances;
import weka.classifiers.trees.J48;

public class DecisionTreeClassifier extends BaseClassifier {
    public DecisionTreeClassifier() {
        modelName = "DecisionTree";
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        classifier = new J48(); //Initialize the J48 classifier
        classifier.buildClassifier( trainingData); //Train the model
        System.out.println("DecisionTree model built successfully.");
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
