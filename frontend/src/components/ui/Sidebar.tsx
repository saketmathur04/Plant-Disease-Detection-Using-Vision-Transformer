import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { 
  Brain, 
  Leaf, 
  Code, 
  Camera, 
  Target, 
  Shield,
  CheckCircle,
  Info,
  Lightbulb
} from "lucide-react";

export function Sidebar() {
  return (
    <aside className="w-80 h-screen bg-gradient-card border-r border-border/50 overflow-y-auto">
      <div className="p-6 space-y-6">
        {/* Project Header */}
        <div className="space-y-4">
          <div className="flex items-center space-x-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-primary">
              <Leaf className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">Plant Disease</h1>
              <p className="text-sm text-plant-primary font-medium">Detection System</p>
            </div>
          </div>
          
          <p className="text-sm text-muted-foreground leading-relaxed">
            Advanced AI-powered plant health diagnostic system using Vision Transformer technology 
            for accurate disease identification and treatment recommendations.
          </p>
        </div>

        <Separator />

        {/* Tech Stack */}
        <Card className="bg-secondary/30 border-border/30 p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Code className="h-4 w-4 text-plant-primary" />
            <h3 className="font-semibold text-foreground">Technology Stack</h3>
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Model:</span>
              <Badge variant="secondary" className="text-xs">ViT Base Patch16</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Framework:</span>
              <Badge variant="secondary" className="text-xs">PyTorch</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">UI:</span>
              <Badge variant="secondary" className="text-xs">React + Tailwind</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Libraries:</span>
              <Badge variant="secondary" className="text-xs">TIMM + Torchvision</Badge>
            </div>
          </div>
        </Card>

        {/* Model Info */}
        <Card className="bg-secondary/30 border-border/30 p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Brain className="h-4 w-4 text-plant-primary" />
            <h3 className="font-semibold text-foreground">Model Details</h3>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex items-start space-x-2">
              <Camera className="h-3 w-3 text-plant-primary mt-1 flex-shrink-0" />
              <div>
                <p className="text-foreground font-medium">Input</p>
                <p className="text-muted-foreground text-xs">224×224 RGB images (JPG, PNG)</p>
              </div>
            </div>
            <div className="flex items-start space-x-2">
              <Target className="h-3 w-3 text-plant-primary mt-1 flex-shrink-0" />
              <div>
                <p className="text-foreground font-medium">Output</p>
                <p className="text-muted-foreground text-xs">Top-3 predictions with confidence</p>
              </div>
            </div>
            <div className="flex items-start space-x-2">
              <Shield className="h-3 w-3 text-plant-primary mt-1 flex-shrink-0" />
              <div>
                <p className="text-foreground font-medium">Coverage</p>
                <p className="text-muted-foreground text-xs">85+ plant diseases across 15+ crops</p>
              </div>
            </div>
          </div>
        </Card>

        {/* Upload Tips */}
        <Card className="bg-plant-primary/5 border-plant-primary/20 p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Lightbulb className="h-4 w-4 text-plant-primary" />
            <h3 className="font-semibold text-foreground">Upload Tips</h3>
          </div>
          <div className="space-y-2">
            {[
              "Use clear, well-lit images",
              "Focus on affected leaf areas",
              "Avoid blurred or dark photos",
              "Include visible symptoms",
              "Single leaf works best"
            ].map((tip, index) => (
              <div key={index} className="flex items-start space-x-2">
                <CheckCircle className="h-3 w-3 text-plant-success mt-0.5 flex-shrink-0" />
                <p className="text-xs text-foreground">{tip}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Supported Crops */}
        <Card className="bg-secondary/30 border-border/30 p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Leaf className="h-4 w-4 text-plant-primary" />
            <h3 className="font-semibold text-foreground">Supported Crops</h3>
          </div>
          <div className="flex flex-wrap gap-1">
            {[
              "Apple", "Cassava", "Cherry", "Chili", "Coffee", "Corn", 
              "Cucumber", "Grape", "Mango", "Peach", "Potato", "Rice", 
              "Soybean", "Tomato", "Wheat"
            ].map((crop) => (
              <Badge 
                key={crop} 
                variant="outline" 
                className="text-xs border-plant-primary/30 text-plant-primary hover:bg-plant-primary/10"
              >
                {crop}
              </Badge>
            ))}
          </div>
        </Card>

        {/* Footer Info */}
        <div className="pt-4 border-t border-border/30">
          <div className="flex items-center space-x-2 mb-2">
            <Info className="h-3 w-3 text-muted-foreground" />
            <p className="text-xs text-muted-foreground">Built with ❤️ for agriculture</p>
          </div>
          <p className="text-xs text-muted-foreground">
            Helping farmers and gardeners protect their crops with AI technology.
          </p>
        </div>
      </div>
    </aside>
  );
}