// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"google.golang.org/genai"
)

// TestImagenConfigType pins the config type image models are defined with.
// The framework deserializes the request's config into it and rejects
// anything else, so the plugin no longer converts configs itself.
func TestImagenConfigType(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		raw         any
		want        genai.GenerateImagesConfig
		expectError bool
	}{
		{
			name: "config struct pointer",
			raw:  &genai.GenerateImagesConfig{NumberOfImages: 2},
			want: genai.GenerateImagesConfig{NumberOfImages: 2},
		},
		{
			name: "config struct value",
			raw:  genai.GenerateImagesConfig{NumberOfImages: 1},
			want: genai.GenerateImagesConfig{NumberOfImages: 1},
		},
		{
			name: "map config",
			raw:  map[string]any{"numberOfImages": 4},
			want: genai.GenerateImagesConfig{NumberOfImages: 4},
		},
		{
			name: "nil config",
			raw:  nil,
			want: genai.GenerateImagesConfig{},
		},
		{
			name:        "another model's config",
			raw:         &genai.GenerateContentConfig{},
			expectError: true,
		},
		{
			name:        "map with a mistyped value",
			raw:         map[string]any{"numberOfImages": "not-a-number"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := base.ConvertToExact[genai.GenerateImagesConfig](tt.raw)
			if (err != nil) != tt.expectError {
				t.Fatalf("ConvertToExact() error = %v, expectError %v", err, tt.expectError)
			}
			if err == nil && got.NumberOfImages != tt.want.NumberOfImages {
				t.Errorf("ConvertToExact() NumberOfImages = %d, want %d", got.NumberOfImages, tt.want.NumberOfImages)
			}
		})
	}
}

func TestTranslateImagenResponse(t *testing.T) {
	t.Parallel()

	resp := &genai.GenerateImagesResponse{
		GeneratedImages: []*genai.GeneratedImage{
			{
				Image: &genai.Image{
					MIMEType:   "image/png",
					ImageBytes: []byte("fake-image-data"),
				},
			},
		},
	}

	res := translateImagenResponse(resp)
	if res.FinishReason != ai.FinishReasonStop {
		t.Errorf("expected finish reason %s, got %s", ai.FinishReasonStop, res.FinishReason)
	}
	if len(res.Message.Content) != 1 {
		t.Fatalf("expected 1 content part, got %d", len(res.Message.Content))
	}
	if res.Message.Content[0].ContentType != "image/png" {
		t.Errorf("expected content type image/png, got %s", res.Message.Content[0].ContentType)
	}
}
