import weka.core.pmml.jaxbbindings.DecisionTree;

public class HelloWeka{
    public static void main(String[] args){
        try {
            ClassificationFramework framework = new ClassificationFramework();

            // Add all classifiers
            framework.addClassifier(new LinearRegressionClassifier());
            framework.addClassifier(new SVMRegressionClassifier());
            framework.addClassifier(new M5PTreeClassifier());
            framework.addClassifier(new RandomForestClassifier());

            // Load processed data into framework
            framework.loadData("processed_data.arff");

            System.out.println("\n=== Starting Model Evaluation ===");
            framework.trainAndEvaluate();
        }
        catch (Exception e) {
            System.err.println("Error in classification: " + e.getMessage());
            e.printStackTrace();
        }
    }
}