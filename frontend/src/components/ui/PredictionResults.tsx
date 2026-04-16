import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { CheckCircle, AlertTriangle, AlertCircle, Leaf } from "lucide-react";

interface Prediction {
  class: string;
  confidence: number;
  isHealthy: boolean;
}

interface PredictionResultsProps {
  prediction: Prediction | null;
  topPredictions: Prediction[];
  isLoading: boolean;
}

export function PredictionResults({ prediction, topPredictions, isLoading }: PredictionResultsProps) {
  if (isLoading) {
    return (
      <Card className="bg-gradient-card shadow-card border-border/50 p-8">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-plant-primary border-t-transparent"></div>
          <p className="text-foreground">Analyzing plant health...</p>
        </div>
      </Card>
    );
  }

  if (!prediction) return null;
  const isNotLeaf = prediction.class.trim().toLowerCase() === "not a leaf image";

  const getStatusIcon = (isHealthy: boolean, confidence: number) => {
    if (isNotLeaf) return <AlertTriangle className="h-6 w-6 text-plant-warning" />;
    if (isHealthy) return <CheckCircle className="h-6 w-6 text-plant-success" />;
    if (confidence > 0.8) return <AlertCircle className="h-6 w-6 text-plant-danger" />;
    return <AlertTriangle className="h-6 w-6 text-plant-warning" />;
  };

  const getStatusColor = (isHealthy: boolean, confidence: number) => {
    if (isNotLeaf) return "plant-warning";
    if (isHealthy) return "plant-success";
    if (confidence > 0.8) return "plant-danger";
    return "plant-warning";
  };

  return (
    <div className="space-y-6">
      {/* Main Prediction */}
      <Card className="bg-gradient-card shadow-card border-border/50 p-8">
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0">
            {getStatusIcon(prediction.isHealthy, prediction.confidence)}
          </div>
          
          <div className="flex-1 space-y-4">
            <div>
              <h3 className="text-2xl font-bold text-foreground mb-2">
                Diagnosis Results
              </h3>
              <div className="flex items-center space-x-3">
                <Badge 
                  variant="secondary"
                  className={`text-${getStatusColor(prediction.isHealthy, prediction.confidence)}-foreground bg-${getStatusColor(prediction.isHealthy, prediction.confidence)}/10 border-${getStatusColor(prediction.isHealthy, prediction.confidence)}/20`}
                >
                  {prediction.class.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </Badge>
                <span className="text-sm text-muted-foreground">
                  {(prediction.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Confidence Level</span>
                <span className="font-medium text-foreground">{(prediction.confidence * 100).toFixed(1)}%</span>
              </div>
              <Progress 
                value={prediction.confidence * 100} 
                className="h-3"
              />
            </div>

            {isNotLeaf ? (
              <div className="bg-plant-warning/10 border border-plant-warning/20 rounded-lg p-4">
                <p className="text-sm text-plant-warning-foreground">
                  Please upload a clear plant leaf image (selfies and random photos will be rejected).
                </p>
              </div>
            ) : prediction.isHealthy ? (
              <div className="bg-plant-success/10 border border-plant-success/20 rounded-lg p-4">
                <p className="text-sm text-plant-success-foreground">
                  ✅ Great news! Your plant appears to be healthy. Continue with regular care and monitoring.
                </p>
              </div>
            ) : (
              <div className="bg-plant-danger/10 border border-plant-danger/20 rounded-lg p-4">
                <p className="text-sm text-plant-danger-foreground">
                  ⚠️ Disease detected. Consider consulting with an agricultural expert for treatment recommendations.
                </p>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Top 3 Predictions */}
      <Card className="bg-gradient-card shadow-card border-border/50 p-6">
        <h4 className="text-lg font-semibold text-foreground mb-4 flex items-center">
          <Leaf className="h-5 w-5 mr-2 text-plant-primary" />
          Alternative Diagnoses
        </h4>
        <div className="space-y-3">
          {topPredictions.map((pred, index) => (
            <div 
              key={index}
              className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 border border-border/30"
            >
              <div className="flex items-center space-x-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-plant-primary/20 flex items-center justify-center">
                  <span className="text-xs font-medium text-plant-primary">{index + 1}</span>
                </div>
                <span className="text-sm font-medium text-foreground">
                  {pred.class.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </span>
              </div>
              <div className="text-right">
                <span className="text-sm text-muted-foreground">
                  {(pred.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}