using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
namespace ApplePricePrediction
{
    public class Program
    {
       
        static readonly string _trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "ApplePrice.zip");
        public static void Main(string[] args)
        {
            MLContext mLContext = new MLContext(seed: 0);
            var model = TrainModel(mLContext, _trainingDataPath);
            PredictSingle(mLContext, model);
        
        }

        public static ITransformer TrainModel(MLContext mLContext, string dataPath)
        {
            IDataView dataView = mLContext.Data.LoadFromTextFile<ApplePrice>(dataPath, hasHeader: true, separatorChar: ',');
            var pipeline = mLContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Price")
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "LocationEncoded", inputColumnName: "Location"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ColorEncoded", inputColumnName: "Color"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "NumberEncoded", inputColumnName: "Number"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SizeEncoded", inputColumnName: "Size"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mLContext.Transforms.Concatenate("Features","LocationEncoded","ColorEncoded","NumberEncoded","SizeEncoded","PaymentTypeEncoded","Price"))
                .Append(mLContext.Regression.Trainers.FastTree())
                ;
            var model = pipeline.Fit(dataView);
            return model;
            
        }
        
        private static void PredictSingle(MLContext mLContext, ITransformer model)
        {
            var predictionFunction = mLContext.Model.CreatePredictionEngine<ApplePrice, ApplePrediction>(model);
            IDataView dataView = mLContext.Data.LoadFromTextFile<ApplePrice>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mLContext.Regression.Evaluate(predictions, "Label", "Score");

            //Input values
            var pricingSample = new ApplePrice()
            {
                Location = "Florida",
                Color = "Red",
                Number = 2,
                Size = 1,
                PaymentType = "Cash",
                Price = 0 // To be predicted
           

        };
            var prediction = predictionFunction.Predict(pricingSample);


            //This area is just for fun, to humanize the output
            var currency = "Dollars";
            if(pricingSample.Location == "Jos")
            {
                prediction.Price = prediction.Price * 365;
                currency = "Naira";
            }
            var size = "";
            
            if (pricingSample.Size== 1)
            {
                 size = "Big";
            }
            else if(pricingSample.Size==2)
            {
                size = "Small";
            }
            var item = "Apple";
            if (pricingSample.Number > 2)
                item = "Apples";
           
            //Humanizer ends

           
            //The console output.
            //Very long to make output appear neat
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine($"---------------------APPLE PRICE PREDICTION-----------------");
            Console.WriteLine();
            Console.WriteLine($"-----------INPUTS-----------");
            Console.WriteLine();
            Console.WriteLine($" Place of Purchase: {pricingSample.Location}");
            Console.WriteLine($" How many: {pricingSample.Number}");
            Console.WriteLine($" Color: {pricingSample.Color}");
            Console.WriteLine($" Size: {size}");
            Console.WriteLine($" Payment Method: {pricingSample.PaymentType}");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine($"-----------------------RESULTS------------------------------");
            Console.WriteLine();
            Console.WriteLine($"{pricingSample.Number} {size}  {item} will cost around {prediction.Price} {currency} in {pricingSample.Location} when paid with {pricingSample.PaymentType}");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("-----Results Error Evaluation-------");
            Console.WriteLine();
            Console.WriteLine($" Supposed Price: 164.25 {currency} ");
            Console.WriteLine($" RS Score: {metrics.RSquared:0.##}");
            Console.WriteLine($" RMS Error: {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine($"-----------BY MUAYYAT BILLAH-------------");
            Console.WriteLine();
            Console.WriteLine($"http://github.com/muayyat/ApplePricePrediction");
            Console.WriteLine($"http://facebook.com/billah.muayyat");
            Console.WriteLine($"http://linkedin.com/in/muayyat");
            Console.WriteLine($"http://twitter.com/muayyat_billah");
            Console.WriteLine($"http://medium.com/muayyat");
            Console.WriteLine();
            Console.WriteLine();

        }
        ///   public static IWebHostBuilder CreateWebHostBuilder(string[] args) =>
        //     WebHost.CreateDefaultBuilder(args)
        //            .UseStartup<Startup>();
    }
}
