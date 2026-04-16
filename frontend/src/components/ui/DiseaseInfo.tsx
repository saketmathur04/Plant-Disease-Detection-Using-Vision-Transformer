import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { 
  ChevronDown, 
  AlertTriangle, 
  Shield, 
  Droplets, 
  Sun,
  Bug,
  Leaf,
  Calendar,
  MapPin
} from "lucide-react";

interface DiseaseInfoProps {
  prediction: {
    class: string;
    confidence: number;
    isHealthy: boolean;
  } | null;
}

export function DiseaseInfo({ prediction }: DiseaseInfoProps) {
  const [isDiseaseInfoOpen, setIsDiseaseInfoOpen] = useState(false);
  const [isPreventionOpen, setIsPreventionOpen] = useState(false);

  if (!prediction) return null;

  // Extract plant and condition from class name
  const [plant, condition] = prediction.class.split('__');
  const plantName = plant.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  const conditionName = condition?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) || 'Unknown';

  // Mock disease data - in real app, this would come from a database
  const diseaseData = {
    symptoms: prediction.isHealthy 
      ? ["Vibrant green color", "Normal leaf structure", "No visible spots or lesions", "Healthy growth pattern"]
      : ["Leaf discoloration", "Spots or lesions", "Wilting or curling", "Unusual growth patterns"],
    
    causes: prediction.isHealthy
      ? ["Proper nutrition", "Adequate watering", "Good air circulation", "Optimal growing conditions"]
      : ["Fungal infection", "Bacterial pathogen", "Environmental stress", "Nutrient deficiency"],
    
    prevention: prediction.isHealthy
      ? [
          "Continue current care routine",
          "Monitor regularly for changes",
          "Maintain proper spacing",
          "Ensure good drainage"
        ]
      : [
          "Remove affected plant parts",
          "Improve air circulation",
          "Adjust watering schedule",
          "Apply appropriate fungicide",
          "Quarantine infected plants"
        ],
    
    treatment: prediction.isHealthy
      ? [
          "Regular monitoring",
          "Maintain optimal conditions",
          "Preventive care measures"
        ]
      : [
          "Immediate isolation",
          "Targeted treatment application",
          "Environmental adjustments",
          "Follow-up monitoring"
        ]
  };

  return (
    <div className="space-y-4">
      {/* Disease Information */}
      <Collapsible open={isDiseaseInfoOpen} onOpenChange={setIsDiseaseInfoOpen}>
        <Card className="bg-gradient-card shadow-card border-border/50">
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className="w-full justify-between p-6 h-auto hover:bg-secondary/30"
            >
              <div className="flex items-center space-x-3">
                <AlertTriangle className="h-5 w-5 text-plant-primary" />
                <div className="text-left">
                  <h3 className="font-semibold text-foreground">Disease Information</h3>
                  <p className="text-sm text-muted-foreground">
                    Learn about {conditionName} in {plantName}
                  </p>
                </div>
              </div>
              <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${isDiseaseInfoOpen ? 'rotate-180' : ''}`} />
            </Button>
          </CollapsibleTrigger>
          
          <CollapsibleContent>
            <div className="px-6 pb-6 space-y-4">
              {/* Plant & Condition Info */}
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center space-x-2">
                  <Leaf className="h-4 w-4 text-plant-primary" />
                  <div>
                    <p className="text-xs text-muted-foreground">Plant Type</p>
                    <p className="text-sm font-medium text-foreground">{plantName}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Bug className="h-4 w-4 text-plant-warning" />
                  <div>
                    <p className="text-xs text-muted-foreground">Condition</p>
                    <p className="text-sm font-medium text-foreground">{conditionName}</p>
                  </div>
                </div>
              </div>

              {/* Symptoms */}
              <div className="space-y-2">
                <h4 className="font-medium text-foreground flex items-center">
                  <AlertTriangle className="h-4 w-4 mr-2 text-plant-warning" />
                  Common Symptoms
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {diseaseData.symptoms.map((symptom, index) => (
                    <div key={index} className="flex items-center space-x-2 text-sm">
                      <div className="w-1.5 h-1.5 rounded-full bg-plant-primary" />
                      <span className="text-muted-foreground">{symptom}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Causes */}
              <div className="space-y-2">
                <h4 className="font-medium text-foreground flex items-center">
                  <MapPin className="h-4 w-4 mr-2 text-plant-danger" />
                  Primary Causes
                </h4>
                <div className="flex flex-wrap gap-2">
                  {diseaseData.causes.map((cause, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      {cause}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Prevention & Treatment */}
      <Collapsible open={isPreventionOpen} onOpenChange={setIsPreventionOpen}>
        <Card className="bg-gradient-card shadow-card border-border/50">
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className="w-full justify-between p-6 h-auto hover:bg-secondary/30"
            >
              <div className="flex items-center space-x-3">
                <Shield className="h-5 w-5 text-plant-success" />
                <div className="text-left">
                  <h3 className="font-semibold text-foreground">Prevention & Treatment</h3>
                  <p className="text-sm text-muted-foreground">
                    Action steps and preventive measures
                  </p>
                </div>
              </div>
              <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${isPreventionOpen ? 'rotate-180' : ''}`} />
            </Button>
          </CollapsibleTrigger>
          
          <CollapsibleContent>
            <div className="px-6 pb-6 space-y-4">
              {/* Prevention */}
              <div className="space-y-2">
                <h4 className="font-medium text-foreground flex items-center">
                  <Shield className="h-4 w-4 mr-2 text-plant-success" />
                  Prevention Measures
                </h4>
                <div className="space-y-2">
                  {diseaseData.prevention.map((measure, index) => (
                    <div key={index} className="flex items-start space-x-3 p-3 bg-plant-success/5 border border-plant-success/20 rounded-lg">
                      <div className="w-5 h-5 rounded-full bg-plant-success/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-xs font-medium text-plant-success">{index + 1}</span>
                      </div>
                      <p className="text-sm text-foreground">{measure}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Treatment */}
              <div className="space-y-2">
                <h4 className="font-medium text-foreground flex items-center">
                  <Droplets className="h-4 w-4 mr-2 text-plant-primary" />
                  Treatment Options
                </h4>
                <div className="space-y-2">
                  {diseaseData.treatment.map((treatment, index) => (
                    <div key={index} className="flex items-start space-x-3 p-3 bg-plant-primary/5 border border-plant-primary/20 rounded-lg">
                      <Calendar className="h-4 w-4 text-plant-primary flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-foreground">{treatment}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Environmental Conditions */}
              <div className="bg-secondary/30 border border-border/30 rounded-lg p-4">
                <h4 className="font-medium text-foreground flex items-center mb-3">
                  <Sun className="h-4 w-4 mr-2 text-plant-warning" />
                  Optimal Growing Conditions
                </h4>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-muted-foreground">Temperature</p>
                    <p className="text-foreground font-medium">18-25°C (65-77°F)</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Humidity</p>
                    <p className="text-foreground font-medium">50-70%</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Light</p>
                    <p className="text-foreground font-medium">6-8 hours/day</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Watering</p>
                    <p className="text-foreground font-medium">Regular, well-drained</p>
                  </div>
                </div>
              </div>
            </div>
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </div>
  );
}