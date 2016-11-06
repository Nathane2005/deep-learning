package nl.vincent.networks;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import nl.vincent.rl.utils.DataUtil;

public class FullyConnected {
	private MultiLayerNetwork model;
	private int inputs, outputs;
	
	public static enum OuputType {
		SOFTMAX("softmax"), LINEAR("identity");
		
		private String value;
		private OuputType(String value) {
			this.value = value;
		}
		
		public String getValue() {
			return value;
		}
	}
	
	public FullyConnected(int inputs, int outputs, int[] hiddenLayers, double learningRate, OuputType outputType) {
		this.inputs = inputs;
		this.outputs = outputs;
		MultiLayerConfiguration conf = null;
		if (hiddenLayers.length == 0) {
	        conf = new NeuralNetConfiguration.Builder()
	                .iterations(1)
	                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	                .learningRate(learningRate)
	                .biasLearningRate(learningRate)
	                .updater(Updater.RMSPROP)
	                .list()
	                .layer(0, new OutputLayer.Builder(LossFunction.MSE)
	                        .weightInit(WeightInit.XAVIER)
	                        .activation(outputType.value).weightInit(WeightInit.XAVIER)
	                        .nIn(inputs).nOut(outputs).build())
	                .pretrain(false).backprop(true).build();
		} else {
			NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
	                .iterations(1)
	                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	                .learningRate(learningRate)
	                .updater(Updater.RMSPROP)
	                .list()
	                .layer(0, new DenseLayer.Builder().nIn(inputs).nOut(hiddenLayers[0])
	                        .weightInit(WeightInit.XAVIER)
	                        .activation("relu")
	                        .biasLearningRate(learningRate)
	                        .biasInit(0)
	                        .build());
	        for (int i = 0 ; i < hiddenLayers.length - 1 ; i++) {
	        	builder.layer(i+1, new DenseLayer.Builder().nIn(hiddenLayers[i]).nOut(hiddenLayers[i+1])
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .biasLearningRate(learningRate)
                        .biasInit(0)
                        .build());
	        }
	        conf =  builder.layer(hiddenLayers.length, new OutputLayer.Builder(LossFunction.MEAN_ABSOLUTE_ERROR)
	                        .weightInit(WeightInit.XAVIER)
	                        .biasLearningRate(learningRate)
	                        .biasInit(0)
	                        .activation(outputType.value).weightInit(WeightInit.XAVIER)
	                        .nIn(hiddenLayers[hiddenLayers.length - 1]).nOut(outputs).build())
	                .pretrain(false).backprop(true).build();
		}
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        this.model = model;
	}
	
	/**
	 * Train the network on an array of input size with a state
	 * @param input
	 * @return
	 */
	public double[] predict(double[] input) {
        INDArray inputVector = DataUtil.toINDArray(input);
        INDArray resultVector = model.output(inputVector);
        double[] result = fromINDArrayVector(resultVector);
		return result;
	}
		
	public void train(double[] input, double[] output) {
		INDArray inputVector = DataUtil.toINDArray(input);
		INDArray outputVector = DataUtil.toINDArray(output);
		model.fit(inputVector, outputVector);
	}
	
	public void train(ArrayList<double[]> input, ArrayList<double[]> output) {
		INDArray inputVector = DataUtil.toINDArray(input);
		INDArray outputVector = DataUtil.toINDArray(output);
		model.fit(inputVector, outputVector);
	}
	
	public void train(DataSet data, int iterations) {
		for (int i=0; i<iterations; i++) {
			model.fit(data);
			System.out.println(model.score());
		}
	}
	
	public double[] fromINDArrayVector(INDArray indArray) {
        double[] result = new double[outputs];
		for (int i = 0; i < outputs; i++) {
			result[i] = indArray.getDouble(i);
		}
		return result;
	}
	
    public Gradient gradient(double[] input, double[] labels) {
        model.setInput(DataUtil.toINDArray(input));
        model.setLabels(DataUtil.toINDArray(labels));
        model.computeGradientAndScore();
        return model.gradient();
    }

    public void loadModel(String fileName) throws IOException {
    	model = ModelSerializer.restoreMultiLayerNetwork(fileName);
    }
    
    public MultiLayerNetwork getModel() {
    	return this.model;
    }

    public void applyGradient(Gradient gradient, int batchSize) {
    	model.getUpdater().update(model, gradient, 1, batchSize);
	    model.params().subi(gradient.gradient());
    }
	
	public void saveModel(String path) throws IOException {
		//Save the model
	    File locationToSave = new File(path);      //Where to save the network. Note: the file is in .zip format - can be opened externally
	    boolean saveUpdater = true;                                     //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
	    ModelSerializer.writeModel(this.model, locationToSave, saveUpdater);
	}
}
