import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.SelectedTag;

public class SVMRegressionClassifier extends BaseClassifier {
    public SVMRegressionClassifier() {
        modelName = "SVM Regression";
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        System.out.println("Building SVM Regression classifier...");

        SMOreg svm = new SMOreg();

        // Setting parameters exactly as shown in the GUI
        svm.setBatchSize("100");
        svm.setC(1.0);  // complexity parameter
        svm.setDebug(false);
        svm.setDoNotCheckCapabilities(false);

        // Set filter type to Normalize training data
        svm.setFilterType(new SelectedTag(SMOreg.FILTER_NORMALIZE, SMOreg.TAGS_FILTER));

        // Configure Polynomial Kernel
        PolyKernel polyKernel = new PolyKernel();
        polyKernel.setCacheSize(250007);
        polyKernel.setExponent(1.0);
        svm.setKernel(polyKernel);

        svm.setNumDecimalPlaces(2);  // number of decimal places set to 2

        // Set RegSMOImproved optimizer with parameters
        String[] regOptimizerOptions = {"-T", "0.001", "-V", "-P", "1.0E-12"};
        svm.getRegOptimizer().setOptions(regOptimizerOptions);

        // Alternative method using options string for complete configuration
        String[] options = new String[] {
                "-C", "1.0",                    // complexity parameter
                "-N", "2",                      // normalize
                "-I", "RegSMOImproved -T 0.001 -V -P 1.0E-12",  // optimizer
                "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007" // kernel
        };
        svm.setOptions(options);

        // Build the classifier
        svm.buildClassifier(trainingData);

        // Store the classifier
        classifier = svm;
        System.out.println("SVM Regression model built successfully.\n");

        // Print model details if available
        if (svm.toString() != null && !svm.toString().isEmpty()) {
            System.out.println("\nModel details:");
            System.out.println(svm.toString());
        }
    }
}