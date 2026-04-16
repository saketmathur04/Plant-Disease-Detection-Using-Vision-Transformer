import { useState, useCallback } from "react";
import { Upload, Image as ImageIcon, X } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  selectedImage: string | null;
  onClearImage: () => void;
}

export function ImageUpload({ onImageSelect, selectedImage, onClearImage }: ImageUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        onImageSelect(file);
      }
    }
  }, [onImageSelect]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onImageSelect(files[0]);
    }
  }, [onImageSelect]);

  if (selectedImage) {
    return (
      <Card className="relative overflow-hidden bg-gradient-card shadow-card border-border/50">
        <div className="relative">
          <img 
            src={selectedImage} 
            alt="Selected plant leaf" 
            className="w-full h-96 object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent" />
          <Button
            onClick={onClearImage}
            variant="destructive"
            size="sm"
            className="absolute top-4 right-4 shadow-button"
          >
            <X className="h-4 w-4" />
          </Button>
          <div className="absolute bottom-4 left-4 text-foreground">
            <p className="text-sm font-medium">Image uploaded successfully</p>
            <p className="text-xs text-muted-foreground">Click 'Analyze Plant Health' to scan for diseases</p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card 
      className={`relative overflow-hidden transition-all duration-300 cursor-pointer group shadow-card
        ${isDragOver 
          ? 'border-plant-primary bg-plant-primary/5 shadow-glow' 
          : 'border-border/50 hover:border-plant-primary/50 bg-gradient-card'
        }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="p-12 text-center space-y-6">
        <div className={`inline-flex h-20 w-20 items-center justify-center rounded-full transition-all duration-300
          ${isDragOver ? 'bg-plant-primary/20 text-plant-primary' : 'bg-secondary text-muted-foreground group-hover:text-plant-primary group-hover:bg-plant-primary/10'}`}>
          <Upload className="h-8 w-8" />
        </div>
        
        <div className="space-y-2">
          <h3 className="text-xl font-semibold text-foreground">
            Upload Plant Image
          </h3>
          <p className="text-muted-foreground">
            Drag and drop your plant leaf image here, or click to browse
          </p>
          <p className="text-xs text-muted-foreground">
            Supports JPG, PNG, JPEG formats
          </p>
        </div>

        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="absolute inset-0 opacity-0 cursor-pointer"
        />

        <Button className="bg-gradient-primary hover:shadow-button transition-all duration-300">
          <ImageIcon className="h-4 w-4 mr-2" />
          Choose Image
        </Button>
      </div>
    </Card>
  );
}