using System.Text;
using System.Text.Json;
using HealthPredictMVC.Models;

namespace HealthPredictMVC.Services
{
    public interface IPredictionService
    {
        Task<PredictionResult> PredictDiabetesAsync(DiabetesInput input);
        Task<PredictionResult> PredictHeartDiseaseAsync(HeartDiseaseInput input);
        Task<PredictionResult> PredictParkinsonsAsync(ParkinsonsInput input);
        Task<PredictionResult> PredictLiverAsync(LiverInput input);
        Task<PredictionResult> PredictKidneyAsync(KidneyInput input);
        Task<PredictionResult> PredictPancreaticAsync(PancreaticInput input);
        Task<bool> CheckHealthAsync();
    }

    public class PredictionService : IPredictionService
    {
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly ILogger<PredictionService> _logger;
        private readonly JsonSerializerOptions _jsonOptions;

        public PredictionService(IHttpClientFactory httpClientFactory, ILogger<PredictionService> logger)
        {
            _httpClientFactory = httpClientFactory;
            _logger = logger;
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
        }

        private HttpClient GetClient()
        {
            return _httpClientFactory.CreateClient("FlaskAPI");
        }

        public async Task<bool> CheckHealthAsync()
        {
            try
            {
                var client = GetClient();
                var response = await client.GetAsync("/api/health");
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Health check failed");
                return false;
            }
        }

        public async Task<PredictionResult> PredictDiabetesAsync(DiabetesInput input)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, object>
                {
                    { "Pregnancies", input.Pregnancies },
                    { "Glucose", input.Glucose },
                    { "BloodPressure", input.BloodPressure },
                    { "SkinThickness", input.SkinThickness },
                    { "Insulin", input.Insulin },
                    { "BMI", input.BMI },
                    { "DiabetesPedigreeFunction", input.DiabetesPedigreeFunction },
                    { "Age", input.Age }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                _logger.LogInformation("Sending diabetes prediction request: {Json}", json);

                var response = await client.PostAsync("/api/diabetes/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                _logger.LogInformation("Received response: {Response}", responseContent);

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<PredictionResult>(responseContent, _jsonOptions);
                    return result ?? new PredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    return new PredictionResult
                    {
                        Success = false,
                        Error = $"API returned status code: {response.StatusCode}"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting diabetes");
                return new PredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }

        public async Task<PredictionResult> PredictHeartDiseaseAsync(HeartDiseaseInput input)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, object>
                {
                    { "Age", input.Age },
                    { "Sex", input.Sex },
                    { "Chest pain type", input.ChestPainType },
                    { "BP", input.BP },
                    { "Cholesterol", input.Cholesterol },
                    { "FBS over 120", input.FBSOver120 },
                    { "EKG results", input.EKGResults },
                    { "Max HR", input.MaxHR },
                    { "Exercise angina", input.ExerciseAngina },
                    { "ST depression", input.STDepression },
                    { "Slope of ST", input.SlopeOfST },
                    { "Number of vessels fluro", input.NumberOfVessels },
                    { "Thallium", input.Thallium }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await client.PostAsync("/api/heart/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<PredictionResult>(responseContent, _jsonOptions);
                    return result ?? new PredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    return new PredictionResult
                    {
                        Success = false,
                        Error = $"API returned status code: {response.StatusCode}"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting heart disease");
                return new PredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }

        public async Task<PredictionResult> PredictParkinsonsAsync(ParkinsonsInput input)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, object>
                {
                    { "MDVP:Fo(Hz)", input.MDVP_Fo },
                    { "MDVP:Fhi(Hz)", input.MDVP_Fhi },
                    { "MDVP:Flo(Hz)", input.MDVP_Flo },
                    { "MDVP:Jitter(%)", input.MDVP_Jitter_Percent },
                    { "MDVP:Jitter(Abs)", input.MDVP_Jitter_Abs },
                    { "MDVP:RAP", input.MDVP_RAP },
                    { "MDVP:PPQ", input.MDVP_PPQ },
                    { "Jitter:DDP", input.Jitter_DDP },
                    { "MDVP:Shimmer", input.MDVP_Shimmer },
                    { "MDVP:Shimmer(dB)", input.MDVP_Shimmer_dB },
                    { "Shimmer:APQ3", input.Shimmer_APQ3 },
                    { "Shimmer:APQ5", input.Shimmer_APQ5 },
                    { "MDVP:APQ", input.MDVP_APQ },
                    { "Shimmer:DDA", input.Shimmer_DDA },
                    { "NHR", input.NHR },
                    { "HNR", input.HNR },
                    { "RPDE", input.RPDE },
                    { "DFA", input.DFA },
                    { "spread1", input.Spread1 },
                    { "spread2", input.Spread2 },
                    { "D2", input.D2 },
                    { "PPE", input.PPE }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await client.PostAsync("/api/parkinsons/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<PredictionResult>(responseContent, _jsonOptions);
                    return result ?? new PredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    return new PredictionResult
                    {
                        Success = false,
                        Error = $"API returned status code: {response.StatusCode}"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting Parkinson's");
                return new PredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }

        public async Task<PredictionResult> PredictLiverAsync(LiverInput input)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, object>
                {
                    { "Age", input.Age },
                    { "Gender", input.Gender },
                    { "Total_Bilirubin", input.TotalBilirubin },
                    { "Direct_Bilirubin", input.DirectBilirubin },
                    { "Alkaline_Phosphotase", input.AlkalinePhosphatase },
                    { "Alamine_Aminotransferase", input.ALT },
                    { "Aspartate_Aminotransferase", input.AST },
                    { "Total_Protiens", input.TotalProteins },
                    { "Albumin", input.Albumin },
                    { "Albumin_and_Globulin_Ratio", input.AGRatio }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await client.PostAsync("/api/liver/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<PredictionResult>(responseContent, _jsonOptions);
                    return result ?? new PredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    return new PredictionResult
                    {
                        Success = false,
                        Error = $"API returned status code: {response.StatusCode}"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting liver disease");
                return new PredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }

        public async Task<PredictionResult> PredictKidneyAsync(KidneyInput input)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, object>
                {
                    { "age", input.Age },
                    { "bp", input.BP },
                    { "sg", input.SG },
                    { "al", input.AL },
                    { "su", input.SU },
                    { "rbc", input.RBC },
                    { "pc", input.PC },
                    { "pcc", input.PCC },
                    { "ba", input.BA },
                    { "bgr", input.BGR },
                    { "bu", input.BU },
                    { "sc", input.SC },
                    { "sod", input.SOD },
                    { "pot", input.POT },
                    { "hemo", input.HEMO },
                    { "pcv", input.PCV },
                    { "wc", input.WC },
                    { "rc", input.RC },
                    { "htn", input.HTN },
                    { "dm", input.DM },
                    { "cad", input.CAD },
                    { "appet", input.APPET },
                    { "pe", input.PE },
                    { "ane", input.ANE }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await client.PostAsync("/api/kidney/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<PredictionResult>(responseContent, _jsonOptions);
                    return result ?? new PredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    return new PredictionResult
                    {
                        Success = false,
                        Error = $"API returned status code: {response.StatusCode}"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting kidney disease");
                return new PredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }

        public async Task<PredictionResult> PredictPancreaticAsync(PancreaticInput input)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, object>
                {
                    { "Age", input.Age },
                    { "Gender", input.Gender },
                    { "Smoking_History", input.SmokingHistory },
                    { "Obesity", input.Obesity },
                    { "Diabetes", input.Diabetes },
                    { "Chronic_Pancreatitis", input.ChronicPancreatitis },
                    { "Family_History", input.FamilyHistory },
                    { "Hereditary_Condition", input.HereditaryCondition },
                    { "Jaundice", input.Jaundice },
                    { "Abdominal_Discomfort", input.AbdominalDiscomfort },
                    { "Back_Pain", input.BackPain },
                    { "Weight_Loss", input.WeightLoss },
                    { "Development_of_Type2_Diabetes", input.DevelopmentType2Diabetes },
                    { "Alcohol_Consumption", input.AlcoholConsumption },
                    { "Physical_Activity_Level", input.PhysicalActivityLevel },
                    { "Diet_Processed_Food", input.DietProcessedFood },
                    { "Access_to_Healthcare", input.AccessToHealthcare },
                    { "Urban_vs_Rural", input.UrbanVsRural },
                    { "Economic_Status", input.EconomicStatus }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await client.PostAsync("/api/pancreatic/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<PredictionResult>(responseContent, _jsonOptions);
                    return result ?? new PredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    return new PredictionResult
                    {
                        Success = false,
                        Error = $"API returned status code: {response.StatusCode}"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting pancreatic cancer");
                return new PredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }
    }
}
