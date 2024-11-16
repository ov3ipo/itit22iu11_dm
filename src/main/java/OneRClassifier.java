import weka.core.Instances;

public class OneRClassifier extends BaseClassifier{

    public OneRClassifier(){
        modelName = "OneR";
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception{
        // Implement OneR algorithm here
        System.out.println("demo");
        throw new UnsupportedOperationException("Not implemented yet");
    }


}
