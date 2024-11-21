
public class HelloWeka{
    public static void main(String[] args){
        try{
            ClassificationFramework framework = new ClassificationFramework();

            framework.addClassifier(new OneRClassifier());

            DataLoader loader = new DataLoader("data/training.csv");
            loader.loadData();
            loader.saveToArff("processed_data.arff");

            // Load processed data into framework
            framework.loadData("processed_data.arff");

            // Train and evaluate
            framework.trainAndEvaluate();
        }
        catch (Exception e) {
            System.err.println("Error in classification: " + e.getMessage());
            e.printStackTrace();
        }
    }
}