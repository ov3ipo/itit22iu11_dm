import weka.classifiers.trees.M5P;
import weka.core.Instances;

public class M5PTreeClassifier extends BaseClassifier {
    public M5PTreeClassifier() {
        modelName = "M5P Decision Tree";
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        System.out.println("Building M5P Decision Tree classifier...");

        M5P m5p = new M5P();

        // Setting parameters as shown in the GUI
        m5p.setBatchSize("100");
        m5p.setDebug(false);
        m5p.setDoNotCheckCapabilities(false);
        m5p.setMinNumInstances(4.0);
        m5p.setNumDecimalPlaces(4);
        m5p.setSaveInstances(false);
        m5p.setUnpruned(false);
        m5p.setUseUnsmoothed(false);
        m5p.setBuildRegressionTree(false);

        // Alternative method using options string
        String[] options = new String[] {
                "-M", "4.0",         // minimum number of instances
                "-U",                // unpruned
                "-R",                // build regression tree
                "-L",                // do not smooth predictions
                "-num-decimal-places", "4"  // number of decimal places
        };
        m5p.setOptions(options);

        // Build the classifier
        m5p.buildClassifier(trainingData);

        // Store the classifier
        classifier = m5p;
        System.out.println("M5P Decision Tree model built successfully.\n");

        // Print model details if available
        if (m5p.toString() != null && !m5p.toString().isEmpty()) {
            System.out.println("\nModel details:");
            System.out.println(m5p.toString());
        }
    }
}