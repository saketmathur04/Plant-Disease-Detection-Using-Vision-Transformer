import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PredictionResults } from "@/components/ui/PredictionResults";
import { Sidebar } from "@/components/ui/Sidebar";
import { WorkingPrinciple } from "@/components/ui/WorkingPrinciple";
import { DiseaseInfo } from "@/components/ui/DiseaseInfo";
import { Scan, Upload, X } from "lucide-react";
import { API_BASE_URL } from "../App";

interface Prediction {
  class: string;
  confidence: number;
  isHealthy: boolean;
}

interface BackendPrediction {
  success: boolean;
  predicted_class: string;
  confidence: number;
  top_predictions: Array<{ class: string; confidence: number }>;
  is_leaf?: boolean;
  message: string;
}

function ImageUpload({ onImageSelect, selectedImage, onClearImage }: any) {
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) onImageSelect(files[0]);
  };

  if (selectedImage) {
    return (
      <Card className="relative overflow-hidden bg-gradient-card shadow-card border-border/50">
        <div className="relative">
          <img src={selectedImage} alt="Selected leaf" className="w-full h-96 object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent" />
          <Button onClick={onClearImage} variant="destructive" size="sm" className="absolute top-4 right-4">
            <X className="h-4 w-4" />
          </Button>
          <div className="absolute bottom-4 left-4 text-foreground">
            <p className="text-sm font-medium">Image uploaded successfully</p>
            <p className="text-xs text-muted-foreground">Click 'Analyze Plant Health' to scan</p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className="border-2 border-dashed border-border/50 rounded-lg p-12 text-center bg-gradient-card shadow-card">
      <div className="space-y-6">
        <div className="inline-flex h-20 w-20 items-center justify-center rounded-full bg-secondary text-muted-foreground">
          <Upload className="h-8 w-8" />
        </div>
        <h3 className="text-xl font-semibold text-foreground">Upload Plant Image</h3>
        <p className="text-muted-foreground">Supports JPG, PNG, JPEG formats</p>
        <input type="file" accept="image/*" onChange={handleFileSelect} className="hidden" id="file-upload" />
        <Button onClick={() => document.getElementById("file-upload")?.click()}>
          <Scan className="h-4 w-4 mr-2" />
          Choose Image
        </Button>
      </div>
    </Card>
  );
}

const Index = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [topPredictions, setTopPredictions] = useState<Prediction[]>([]);

  const handleImageSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setSelectedImage(URL.createObjectURL(file));
    setPrediction(null);
    setTopPredictions([]);
  }, []);

  const handleClearImage = useCallback(() => {
    if (selectedImage) URL.revokeObjectURL(selectedImage);
    setSelectedFile(null);
    setSelectedImage(null);
    setPrediction(null);
    setTopPredictions([]);
  }, [selectedImage]);

  const analyzeImage = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, { method: "POST", body: formData });
      const result: BackendPrediction = await response.json();
      if (!result.success) throw new Error(result.message);

      const mainPrediction: Prediction = {
        class: result.predicted_class,
        confidence: result.confidence,
        isHealthy:
          (result.is_leaf === false) ||
          result.predicted_class.toLowerCase().includes("healthy"),
      };

      const topPreds: Prediction[] = result.top_predictions.map(pred => ({
        class: pred.class,
        confidence: pred.confidence,
        isHealthy: pred.class.toLowerCase().includes("healthy"),
      }));

      setPrediction(mainPrediction);
      setTopPredictions(topPreds);
    } catch (error) {
      alert(`Prediction failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <div className="container mx-auto px-6 py-8 max-w-5xl space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold text-foreground">Plant Disease Detection</h1>
            <p className="text-muted-foreground max-w-2xl mx-auto">Upload an image of your plant leaf to get instant AI-powered disease diagnosis.</p>
          </div>

          <ImageUpload onImageSelect={handleImageSelect} selectedImage={selectedImage} onClearImage={handleClearImage} />

          {selectedImage && !isAnalyzing && !prediction && (
            <div className="text-center">
              <Button onClick={analyzeImage} className="bg-gradient-primary px-8 py-3 text-lg">
                <Scan className="h-5 w-5 mr-2" /> Analyze Plant Health
              </Button>
            </div>
          )}

          <PredictionResults prediction={prediction} topPredictions={topPredictions} isLoading={isAnalyzing} />

          {prediction && <DiseaseInfo prediction={prediction} />}
          <WorkingPrinciple />
        </div>
      </main>
    </div>
  );
};

export default Index;
