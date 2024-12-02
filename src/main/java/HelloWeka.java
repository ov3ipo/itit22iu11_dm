public class HelloWeka{
    public static void main(String[] args){
        try {
            ModelFramework framework = new ModelFramework();
            DataLoader loader = new DataLoader("data/test.csv");
            loader.loadData();
            loader.saveToArff("processed_data.arff");

            // Add all classifiers
            framework.addClassifier(new LinearRegression());
            framework.addClassifier(new SVMRegression());
            framework.addClassifier(new M5PTreeClassifier());
            framework.addClassifier(new RandomForest());

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