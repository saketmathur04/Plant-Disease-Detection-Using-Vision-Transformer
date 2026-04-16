import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Upload, 
  Brain, 
  Zap, 
  BarChart3, 
  Target,
  ArrowRight,
  Image as ImageIcon,
  Cpu,
  TrendingUp
} from "lucide-react";

export function WorkingPrinciple() {
  const steps = [
    {
      icon: Upload,
      title: "Image Upload",
      description: "User uploads a clear image of the plant leaf showing symptoms",
      technical: "Image preprocessing: resize to 224×224, normalize RGB values",
      color: "text-blue-400"
    },
    {
      icon: Cpu,
      title: "Preprocessing",
      description: "Image is resized, normalized and converted to tensor format",
      technical: "Apply transforms: Resize, ToTensor, Normalize with ImageNet stats",
      color: "text-purple-400"
    },
    {
      icon: Brain,
      title: "ViT Analysis",
      description: "Vision Transformer processes image patches and generates predictions",
      technical: "ViT splits image into 16×16 patches, applies self-attention mechanism",
      color: "text-plant-primary"
    },
    {
      icon: Zap,
      title: "Softmax Activation",
      description: "Raw logits converted to probability distributions",
      technical: "Apply softmax function: P(class_i) = exp(logit_i) / Σexp(logit_j)",
      color: "text-yellow-400"
    },
    {
      icon: TrendingUp,
      title: "Top-K Predictions",
      description: "Extract top-3 predictions with highest confidence scores",
      technical: "Sort probabilities, select top 3 classes with confidence levels",
      color: "text-orange-400"
    },
    {
      icon: Target,
      title: "Disease Mapping",
      description: "Map class indices to human-readable disease names and recommendations",
      technical: "Index-to-name mapping with disease info and treatment suggestions",
      color: "text-red-400"
    }
  ];

  return (
    <Card className="bg-gradient-card shadow-card border-border/50 p-6">
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold text-foreground flex items-center justify-center">
            <Brain className="h-6 w-6 mr-2 text-plant-primary" />
            How It Works
          </h2>
          <p className="text-muted-foreground">
            Understanding the Vision Transformer (ViT) disease detection pipeline
          </p>
        </div>

        <div className="grid gap-4">
          {steps.map((step, index) => (
            <div key={index} className="relative">
              <Card className="bg-secondary/30 border-border/30 p-4 hover:bg-secondary/50 transition-colors">
                <div className="flex items-start space-x-4">
                  <div className={`flex h-10 w-10 items-center justify-center rounded-lg bg-secondary/50 ${step.color}`}>
                    <step.icon className="h-5 w-5" />
                  </div>
                  
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline" className="text-xs">
                        Step {index + 1}
                      </Badge>
                      <h3 className="font-semibold text-foreground">{step.title}</h3>
                    </div>
                    
                    <p className="text-sm text-muted-foreground">
                      {step.description}
                    </p>
                    
                    <div className="bg-background/50 rounded-md p-2 border border-border/30">
                      <p className="text-xs font-mono text-plant-primary">
                        {step.technical}
                      </p>
                    </div>
                  </div>
                </div>
              </Card>
              
              {index < steps.length - 1 && (
                <div className="flex justify-center my-2">
                  <ArrowRight className="h-4 w-4 text-muted-foreground" />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Key Features */}
        <Card className="bg-plant-primary/5 border-plant-primary/20 p-4">
          <h3 className="font-semibold text-foreground mb-3 flex items-center">
            <BarChart3 className="h-4 w-4 mr-2 text-plant-primary" />
            Key Technical Features
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[
              { label: "Architecture", value: "Vision Transformer (ViT-B/16)" },
              { label: "Input Resolution", value: "224 × 224 pixels" },
              { label: "Patch Size", value: "16 × 16 pixels" },
              { label: "Total Classes", value: "85+ disease types" },
              { label: "Inference Time", value: "< 2 seconds" },
              { label: "Model Size", value: "~330MB parameters" }
            ].map((feature, index) => (
              <div key={index} className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">{feature.label}:</span>
                <Badge variant="secondary" className="text-xs">{feature.value}</Badge>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </Card>
  );
}