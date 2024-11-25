import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class RandomForestClassifier extends BaseClassifier {
    public RandomForestClassifier() {
        modelName = "RandomForest";
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        System.out.println("Building RandomForest classifier...");

        RandomForest rf = new RandomForest();

        // Setting parameters as shown in the GUI
        rf.setBagSizePercent(100);
        rf.setBatchSize("100");
        rf.setBreakTiesRandomly(false);
        rf.setCalcOutOfBag(false);
        rf.setComputeAttributeImportance(false);
        rf.setDebug(false);
        rf.setDoNotCheckCapabilities(false);
        rf.setMaxDepth(0);
        rf.setNumDecimalPlaces(2);
        rf.setNumExecutionSlots(1);
        rf.setNumFeatures(0);
        rf.setNumIterations(100);
        rf.setPrintClassifiers(false);
        rf.setSeed(1);
        rf.setStoreOutOfBagPredictions(false);

        // Build the classifier
        rf.buildClassifier(trainingData);

        // Store the classifier
        classifier = rf;
        System.out.println("RandomForest model built successfully.\n");
    }
}