import weka.core.Instances;
import weka.classifiers.trees.J48;

public class DecisionTreeClassifier extends BaseClassifier {
    public DecisionTreeClassifier() {
        modelName = "DecisionTree";
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        classifier = new J48();

        // Configure J48 parameters
        String[] options = new String[]{
                "-R",          // Use reduced error pruning
                "-N", "3",     // Number of folds for reduced error pruning
                "-M", "5"      // Higher minimum instances per leaf
        };
        ((J48) classifier).setOptions(options);

        // Build the model
        classifier.buildClassifier(trainingData);
        System.out.println("DecisionTree model built successfully.");
    }
}
