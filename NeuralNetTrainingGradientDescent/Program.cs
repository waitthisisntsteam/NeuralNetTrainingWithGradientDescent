using System;

namespace NeuralNetTrainingGradientDescent
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ActivationErorrFormulas activationErorrFormulas = new ActivationErorrFormulas();
            ErrorFunction Error = new ErrorFunction(activationErorrFormulas.MeanSquared, activationErorrFormulas.MeanSquaredD);
            ActivationFunction Activation = new ActivationFunction(activationErorrFormulas.TanH, activationErorrFormulas.TanHD);

            NeuralNetwork network = new NeuralNetwork([Activation], Error, 1, 5, 5, 5, 5, 1);
            network.Randomize(new Random(), -1, 1);

            const int portions = 16;
            const double learningRate = 0.001;
            const double momentum = 0.01;
            const int batchSize = 5;

            double[][] desiredOutputs = new double[portions][];
            double[][] inputs = new double[portions][];

            for (int i = 0; i < portions; i++)
            {
                inputs[i] = [Math.PI / portions * i];

                desiredOutputs[i] = [Math.Sin(inputs[i][0])];
            }

            double error = network.BatchTrain(inputs, desiredOutputs, batchSize, learningRate, momentum);
            double originalError = error;
            double oldError = originalError;
            while (true)
            {
                Console.WriteLine("Starting Error:");
                Console.WriteLine(originalError);
                Console.WriteLine("      ");

                Console.WriteLine("Current Error:");
                error = network.BatchTrain(inputs, desiredOutputs, batchSize, learningRate, momentum);
                Console.Write(error);
                Console.WriteLine();

                Console.WriteLine("Current Outputs:");
                for (int i = 0; i < inputs.Length; i++)
                {
                    var output = network.Compute(inputs[i]);

                    for (int j = 0; j < output.Length; j++)
                    {
                        Console.Write(desiredOutputs[i][0] + "       ");
                        Console.Write(output[j]);
                        Console.WriteLine();
                    }
                }

                oldError = error;
                Console.Clear();
            }
        }
    }
}