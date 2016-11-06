package nl.vincent.rl.datasets;

import org.nd4j.linalg.dataset.DataSet;

import nl.vincent.rl.utils.DataUtil;

/**
 * Simple test dataset
 * @author vincentbons
 */
public class XOR {

	public static DataSet getDataSet() {

		
		DataSet dataSet = new DataSet(DataUtil.toINDArray(getRawInputs()), DataUtil.toINDArray(getRawOutputs()));
		
		return dataSet;
	}
	
	public static double[][] getRawInputs() {
		double[][] inputs = new double[4][2];
		
		// input 0 0, output 0
		inputs[0][0] = 0;
		inputs[0][1] = 0;
		
		// input 1 0, output 1
		inputs[1][0] = 1;
		inputs[1][1] = 0;
		
		// input 0 1, output 1
		inputs[2][0] = 0;
		inputs[2][1] = 1;
		
		// input 1 1, output 0
		inputs[3][0] = 1;
		inputs[3][1] = 1;
		return inputs;
	}
	
	public static double[][] getRawOutputs() {
		double[][] outputs = new double[4][1];
		
		// input 0 0, output 0
		outputs[0][0] = 0;
		
		// input 1 0, output 1
		outputs[1][0] = 1;
		
		// input 0 1, output 1
		outputs[2][0] = 1;
		
		// input 1 1, output 0
		outputs[3][0] = 0;
		return outputs;
	}
}
