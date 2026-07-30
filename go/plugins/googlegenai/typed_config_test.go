// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"google.golang.org/genai"
)

// testClient builds a client that never talks to the API. Constructing the
// actions below only reads the backend off the client config.
func testClient(t *testing.T) *genai.Client {
	t.Helper()
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  "test-api-key",
	})
	if err != nil {
		t.Fatalf("genai.NewClient() error = %v", err)
	}
	return client
}

// validateConfig runs the request's config through the same check the action
// boundary performs: the config schema the model advertises is enforced on
// every call.
func validateConfig(t *testing.T, inputSchema map[string]any, config any) error {
	t.Helper()
	return base.ValidateValue(&ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(ai.NewTextPart("hello"))},
		Config:   config,
	}, inputSchema)
}

// TestModelConfigSchema pins the schema each model advertises for its config,
// now that the framework validates requests against it. Fields hidden from the
// dev UI must still reach the plugin's own checks, which say what to use
// instead, rather than being rejected as unknown properties.
func TestModelConfigSchema(t *testing.T) {
	t.Parallel()
	client := testClient(t)

	gemini := newModel(client, gemini25Flash, GetModelOptions(gemini25Flash, googleAIProvider)).Desc()
	imagen := newModel(client, imagen40Generate001, GetModelOptions(imagen40Generate001, googleAIProvider)).Desc()

	if got := gemini.Metadata["model"].(map[string]any)["customOptions"]; !reflect.DeepEqual(got, geminiConfigSchema) {
		t.Error("gemini customOptions is not the curated GenerateContentConfig schema")
	}
	if got := imagen.Metadata["model"].(map[string]any)["customOptions"]; !reflect.DeepEqual(got, imagenConfigSchema) {
		t.Error("imagen customOptions is not the curated GenerateImagesConfig schema")
	}

	accepted := []struct {
		name   string
		config any
	}{
		{"struct config", genai.GenerateContentConfig{Temperature: genai.Ptr[float32](0.4)}},
		{"pointer config", &genai.GenerateContentConfig{MaxOutputTokens: 100}},
		{"map config", map[string]any{"temperature": 0.4}},
		{"nil config", nil},
		// A typed nil marshals to JSON null, which the config slot tolerates.
		{"typed nil config", (*genai.GenerateContentConfig)(nil)},
		// Hidden from the dev UI, but the plugin pins it to 1 rather than
		// rejecting the field outright.
		{"candidateCount", map[string]any{"candidateCount": 1}},
		// Hidden because Genkit primitives own them: toGeminiRequest answers
		// these with the primitive to use instead.
		{"systemInstruction", map[string]any{"systemInstruction": map[string]any{"parts": []any{map[string]any{"text": "hi"}}}}},
		{"responseMimeType", map[string]any{"responseMimeType": "application/json"}},
		{"config tool functionDeclarations", map[string]any{"tools": []any{map[string]any{"functionDeclarations": []any{map[string]any{"name": "x"}}}}}},
	}
	for _, tt := range accepted {
		t.Run("gemini accepts "+tt.name, func(t *testing.T) {
			if err := validateConfig(t, gemini.InputSchema, tt.config); err != nil {
				t.Errorf("config rejected at the action boundary: %v", err)
			}
		})
	}

	rejected := []struct {
		name        string
		inputSchema map[string]any
		config      any
	}{
		{"mistyped value", gemini.InputSchema, map[string]any{"temperature": "hot"}},
		{"unknown nested field", gemini.InputSchema, map[string]any{"thinkingConfig": map[string]any{"nope": 1}}},
		// Image models advertise the images config, so a chat config's fields
		// are not valid there.
		{"chat config on an image model", imagen.InputSchema, map[string]any{"temperature": 0.4}},
	}
	for _, tt := range rejected {
		t.Run("rejects "+tt.name, func(t *testing.T) {
			if err := validateConfig(t, tt.inputSchema, tt.config); err == nil {
				t.Error("expected the action boundary to reject this config")
			}
		})
	}
}

// TestVeoConfigSchema checks the background model advertises the video config,
// which the framework validates and deserializes for the start function.
func TestVeoConfigSchema(t *testing.T) {
	t.Parallel()
	client := testClient(t)

	desc := newVeoModel(client, veo30Generate001, GetModelOptions(veo30Generate001, googleAIProvider)).Desc()
	if got := desc.Metadata["model"].(map[string]any)["customOptions"]; !reflect.DeepEqual(got, veoConfigSchema) {
		t.Error("veo customOptions is not the curated GenerateVideosConfig schema")
	}
	if err := validateConfig(t, desc.InputSchema, map[string]any{"aspectRatio": "16:9", "durationSeconds": 5}); err != nil {
		t.Errorf("video config rejected at the action boundary: %v", err)
	}
	if err := validateConfig(t, desc.InputSchema, map[string]any{"temperature": 0.4}); err == nil {
		t.Error("expected a chat config to be rejected by a video model")
	}
}

// TestEmbedderConfigSchema covers the embedder's config slot. Options used to
// be read with a pointer type assertion, so a map (what the dev UI and every
// other JSON caller sends) was silently dropped; the framework now
// deserializes it into the typed value the embedder is defined with.
func TestEmbedderConfigSchema(t *testing.T) {
	t.Parallel()
	client := testClient(t)

	opts := GetEmbedderOptions(textembeddinggecko003, vertexAIProvider)
	desc := newEmbedder(client, textembeddinggecko003, &opts).Desc()

	for _, options := range []any{
		genai.EmbedContentConfig{TaskType: "RETRIEVAL_DOCUMENT"},
		&genai.EmbedContentConfig{OutputDimensionality: genai.Ptr[int32](256)},
		map[string]any{"taskType": "RETRIEVAL_QUERY"},
		nil,
	} {
		if err := base.ValidateValue(&ai.EmbedRequest{Options: options}, desc.InputSchema); err != nil {
			t.Errorf("embedder options %#v rejected at the action boundary: %v", options, err)
		}
	}

	if err := base.ValidateValue(&ai.EmbedRequest{
		Options: map[string]any{"taskType": 42},
	}, desc.InputSchema); err == nil {
		t.Error("expected a mistyped taskType to be rejected")
	}
}
