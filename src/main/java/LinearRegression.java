import weka.core.Instances;
import weka.core.SelectedTag;

public class LinearRegression extends Base {
    public LinearRegression() {
        modelName = "LinearRegression";
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        System.out.println("Building LinearRegression classifier...");

        weka.classifiers.functions.LinearRegression lr = new weka.classifiers.functions.LinearRegression();

        // Setting parameters as shown in the GUI
        lr.setAttributeSelectionMethod(new SelectedTag(1, weka.classifiers.functions.LinearRegression.TAGS_SELECTION)); // M5 method
        lr.setBatchSize("100");
        lr.setDebug(false);
        lr.setDoNotCheckCapabilities(false);
        lr.setEliminateColinearAttributes(true);
        lr.setMinimal(false);
        lr.setNumDecimalPlaces(4);
        lr.setOutputAdditionalStats(false);
        lr.setRidge(1.0E-8);
        lr.setUseQRDecomposition(false);

        // Alternative method using options string
        String[] options = new String[] {
                "-S", "1",           // attributeSelectionMethod (M5)
                "-R", "1.0E-8",      // ridge parameter
                "-num-decimal-places", "4"  // number of decimal places
        };
        lr.setOptions(options);

        // Build the classifier
        lr.buildClassifier(trainingData);

        // Store the classifier
        classifier = lr;
        System.out.println("LinearRegression model built successfully.\n");

        // Print model details if available
        if (lr.toString() != null && !lr.toString().isEmpty()) {
            System.out.println("\nModel details:");
            System.out.println(lr.toString());
        }
    }
}