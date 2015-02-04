//
// TemporalSegmentations.cs
//
// Author:
//       Tom Diethe <tom.diethe@bristol.ac.uk>
//
// Copyright (c) 2015 University of Bristol
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

namespace HiddenMarkovModel
{
    using System;
    using System.Linq;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// The Program.
    /// </summary>
    public class Program
    {
        /// <summary>
        /// The entry point of the program, where the program control starts and ends.
        /// </summary>
        /// <param name="args">The command-line arguments.</param>
        public static void Main(string[] args)
        {
            TestHiddenMarkovModel();
            TestBinaryHiddenMarkovModel();
            TestDiscreteHiddenMarkovModel();
        }

        /// <summary>
        /// Tests the hidden markov model.
        /// </summary>
        public static void TestHiddenMarkovModel()
        {
            // fix random seed
            Rand.Restart(12347);

            // model size
            const int T = 100;
            const int K = 2;

            // set hyperparameters
            var probInitPriorObs = Dirichlet.Uniform(K);
            var cptTransPriorObs = Enumerable.Repeat(Dirichlet.Uniform(K), K).ToArray();
            var emitMeanPriorObs = Enumerable.Repeat(Gaussian.FromMeanAndVariance(0, 1000), K).ToArray();
            var emitPrecPriorObs = Enumerable.Repeat(Gamma.FromShapeAndScale(1000, 0.001), K).ToArray();

            // sample model parameters
            var init = probInitPriorObs.Sample().ToArray();
            var trans = new double[K][];
            for (int i = 0; i < K; i++)
            {
                trans[i] = cptTransPriorObs[i].Sample().ToArray();
            }

            var emitMeans = new double[K];
            for (int i = 0; i < K; i++)
            {
                emitMeans[i] = emitMeanPriorObs[i].Sample();
            }

            var emitPrecs = new double[K];
            for (int i = 0; i < K; i++)
            {
                emitPrecs[i] = emitPrecPriorObs[i].Sample();
            }

            // print parameters
            var modelForPrinting = new HiddenMarkovModel(T, K);
            modelForPrinting.SetParameters(init, trans, emitMeans, emitPrecs);
            Console.WriteLine("parameters:");
            modelForPrinting.PrintParameters();
            Console.WriteLine();

            // create distributions for sampling
            var initDist = new Discrete(init);
            var transDist = new Discrete[K];
            for (int i = 0; i < K; i++)
            {
                transDist[i] = new Discrete(trans[i]);
            }

            var emitDist = new Gaussian[K];
            for (int i = 0; i < K; i++)
            {
                emitDist[i] = Gaussian.FromMeanAndPrecision(emitMeans[i], emitPrecs[i]);
            }

            // calculate order of actual states
            var actualStateOrder = ArgSort(emitDist);
            Console.WriteLine("actualStateOrder");
            Console.WriteLine(string.Join(",", actualStateOrder));
            Console.WriteLine();

            // sample state and emission data
            var actualStates = new int[T];
            var emissions = new double[T];
            actualStates[0] = initDist.Sample();
            emissions[0] = emitDist[actualStates[0]].Sample();
            for (int i = 1; i < T; i++)
            {
                actualStates[i] = transDist[actualStates[i - 1]].Sample();
                emissions[i] = emitDist[actualStates[i]].Sample();
            }

            Console.WriteLine("sample data:");
            Console.WriteLine(string.Join(",", actualStates));
            Console.WriteLine();

            // infer model parameters, states and model evidence given priors and emission data
            var model = new HiddenMarkovModel(T, K);
            model.SetPriors(probInitPriorObs, cptTransPriorObs, emitMeanPriorObs, emitPrecPriorObs);
            model.ObserveData(emissions);
            model.InitialiseStatesRandomly();
            model.InferPosteriors();
            Console.WriteLine("model likelihood: " + model.modelEvidencePosterior);
            var mapStatesDistr = model.statesPosterior;
            var mapStates = mapStatesDistr.Select(s => s.GetMode()).ToArray();
            Console.WriteLine();

            // print maximum a priori states
            Console.WriteLine("statesMAP");
            Console.WriteLine(string.Join(",", mapStates));
            Console.WriteLine();

            // print posterior distributions
            Console.WriteLine("posteriors");
            model.PrintPosteriors();
            Console.WriteLine();

            // calculate order of MAP states
            int[] mapStateOrder = ArgSort(model.emitMeanPosterior);
            Console.WriteLine("mapStateOrder");
            Console.WriteLine(string.Join(",", mapStateOrder));
            Console.WriteLine();

            // accuracy of MAP estimates
            int correctStates = actualStates.Where((t, i) => actualStateOrder[t] == mapStateOrder[mapStates[i]]).Count();
            Console.WriteLine("correctStates: " + correctStates + " / " + actualStates.Length);
            Console.WriteLine();

            Console.WriteLine("------------------\n");
        }

        /// <summary>
        /// Tests the binary hidden markov model.
        /// </summary>
        public static void TestBinaryHiddenMarkovModel()
        {
            // fix random seed
            Rand.Restart(12347);

            // model size
            const int T = 100;
            const int K = 2;

            // set hyperparameters
            var probInitPriorObs = Dirichlet.Uniform(K);
            var cptTransPriorObs = Enumerable.Repeat(Dirichlet.Uniform(K), K).ToArray();
            var emitPriorObs = Enumerable.Repeat(new Beta(1, 1), K).ToArray();

            // sample model parameters
            var init = probInitPriorObs.Sample().ToArray();
            var trans = new double[K][];
            for (int i = 0; i < K; i++)
            {
                trans[i] = cptTransPriorObs[i].Sample().ToArray();
            }

            var emit = new double[K];
            for (int i = 0; i < K; i++)
            {
                emit[i] = emitPriorObs[i].Sample();
            }

            // print parameters
            var modelForPrinting = new BinaryHiddenMarkovModel(T, K);
            modelForPrinting.SetParameters(init, trans, emit);
            Console.WriteLine("parameters:");
            modelForPrinting.PrintParameters();
            Console.WriteLine();

            // create distributions for sampling
            var initDist = new Discrete(init);
            var transDist = new Discrete[K];
            for (int i = 0; i < K; i++)
            {
                transDist[i] = new Discrete(trans[i]);
            }
            
            var emitDist = new Bernoulli[K];
            for (int i = 0; i < K; i++)
            {
                emitDist[i] = new Bernoulli(emit[i]);
            }

            // calculate order of actual states
            var actualStateOrder = ArgSort(emitDist);
            Console.WriteLine("actualStateOrder");
            Console.WriteLine(string.Join(",", actualStateOrder));
            Console.WriteLine();

            // sample state and emission data
            var actualStates = new int[T];
            var emissions = new bool[T];
            actualStates[0] = initDist.Sample();
            emissions[0] = emitDist[actualStates[0]].Sample();
            for (int i = 1; i < T; i++)
            {
                actualStates[i] = transDist[actualStates[i - 1]].Sample();
                emissions[i] = emitDist[actualStates[i]].Sample();
            }

            Console.WriteLine("sample data:");
            Console.WriteLine(string.Join(",", actualStates));
            Console.WriteLine();

            // infer model parameters, states and model evidence given priors and emission data
            var model = new BinaryHiddenMarkovModel(T, K);
            model.SetPriors(probInitPriorObs, cptTransPriorObs, emitPriorObs);
            model.ObserveData(emissions);
            model.InitialiseStatesRandomly();
            model.InferPosteriors();
            Console.WriteLine("model likelihood: " + model.modelEvidencePosterior);
            var mapStatesDistr = model.statesPosterior;
            var mapStates = mapStatesDistr.Select(s => s.GetMode()).ToArray();
            Console.WriteLine();

            // print maximum a priori states
            Console.WriteLine("statesMAP");
            Console.WriteLine(string.Join(",", mapStates));
            Console.WriteLine();

            // print posterior distributions
            Console.WriteLine("posteriors");
            model.PrintPosteriors();
            Console.WriteLine();

            // calculate order of MAP states
            var mapStateOrder = ArgSort(model.emitPosterior);
            Console.WriteLine("mapStateOrder");
            Console.WriteLine(string.Join(",", mapStateOrder));
            Console.WriteLine();

            // accuracy of MAP estimates
            int correctStates = actualStates.Where((t, i) => actualStateOrder[t] == mapStateOrder[mapStates[i]]).Count();
            Console.WriteLine("correctStates: " + correctStates + " / " + actualStates.Length);
            Console.WriteLine();

            Console.WriteLine("------------------\n");
        }

        /// <summary>
        /// Tests the discrete hidden markov model.
        /// </summary>
        public static void TestDiscreteHiddenMarkovModel()
        {
            // fix random seed
            Rand.Restart(12347);

            // model size
            const int T = 100;
            const int K = 2;
            const int E = 5;

            // set hyperparameters
            var probInitPriorObs = Dirichlet.Uniform(K);
            var cptTransPriorObs = Enumerable.Repeat(Dirichlet.Uniform(K), K).ToArray();
            var emitPriorObs = Enumerable.Repeat(Dirichlet.Uniform(E), K).ToArray();

            // sample model parameters
            var init = probInitPriorObs.Sample().ToArray();
            var trans = new double[K][];
            for (int i = 0; i < K; i++)
            {
                trans[i] = cptTransPriorObs[i].Sample().ToArray();
            }

            var emit = new Vector[K];
            for (int i = 0; i < K; i++)
            {
                emit[i] = emitPriorObs[i].Sample();
            }

            // print parameters
            var modelForPrinting = new DiscreteHiddenMarkovModel(T, K, E);
            modelForPrinting.SetParameters(init, trans, emit);
            Console.WriteLine("parameters:");
            modelForPrinting.PrintParameters();
            Console.WriteLine();

            // create distributions for sampling
            var initDist = new Discrete(init);
            var transDist = new Discrete[K];
            for (int i = 0; i < K; i++)
            {
                transDist[i] = new Discrete(trans[i]);
            }

            var emitDist = new Discrete[K];
            for (int i = 0; i < K; i++)
            {
                emitDist[i] = new Discrete(emit[i]);
            }

            // calculate order of actual states
            var actualStateOrder = ArgSort(emitDist);
            Console.WriteLine("actualStateOrder");
            Console.WriteLine(string.Join(",", actualStateOrder));
            Console.WriteLine();

            // sample state and emission data
            var actualStates = new int[T];
            var emissions = new int[T];
            actualStates[0] = initDist.Sample();
            emissions[0] = emitDist[actualStates[0]].Sample();
            for (int i = 1; i < T; i++)
            {
                actualStates[i] = transDist[actualStates[i - 1]].Sample();
                emissions[i] = emitDist[actualStates[i]].Sample();
            }

            Console.WriteLine("sample data:");
            Console.WriteLine(string.Join(",", actualStates));
            Console.WriteLine();

            // infer model parameters, states and model evidence given priors and emission data
            var model = new DiscreteHiddenMarkovModel(T, K, E);
            model.SetPriors(probInitPriorObs, cptTransPriorObs, emitPriorObs);
            model.ObserveData(emissions);
            model.InitialiseStatesRandomly();
            model.InferPosteriors();
            Console.WriteLine("model likelihood: " + model.modelEvidencePosterior);
            var mapStatesDistr = model.statesPosterior;
            var mapStates = mapStatesDistr.Select(s => s.GetMode()).ToArray();
            Console.WriteLine();

            // print maximum a priori states
            Console.WriteLine("statesMAP");
            Console.WriteLine(string.Join(",", mapStates));
            Console.WriteLine();

            // print posterior distributions
            Console.WriteLine("posteriors");
            model.PrintPosteriors();
            Console.WriteLine();
            /*
            // calculate order of MAP states
            var mapStateOrder = ArgSort(model.emitPosterior);
            Console.WriteLine("mapStateOrder");
            Console.WriteLine(string.Join(",", mapStateOrder));
            Console.WriteLine();

            // accuracy of MAP estimates
            int correctStates = actualStates.Where((t, i) => actualStateOrder[t] == mapStateOrder[mapStates[i]]).Count();
            Console.WriteLine("correctStates: " + correctStates + " / " + actualStates.Length);
            Console.WriteLine();
            */
            Console.WriteLine("------------------\n");
        }

        /// <summary>
        /// Returns the indices of sorting.
        /// </summary>
        /// <typeparam name="TDistribution">The type of the distribution.</typeparam>
        /// <param name="emitMeanPosterior">Emit mean posterior.</param>
        /// <returns>
        /// The indices.
        /// </returns>
        public static int[] ArgSort<TDistribution>(TDistribution[] emitMeanPosterior)
            where TDistribution : CanGetMean<double>
        {
            int[] order = new int[emitMeanPosterior.Length];
            for (int i = 0; i < emitMeanPosterior.Length; i++)
            {
                foreach (var dist in emitMeanPosterior)
                {
                    if (emitMeanPosterior[i].GetMean() > dist.GetMean())
                    {
                        order[i]++;
                    }
                }
            }
            
            return order;
        }
    }
}