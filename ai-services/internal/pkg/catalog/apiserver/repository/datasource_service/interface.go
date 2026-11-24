package datasourceservice

import (
	"context"
	"fmt"
	"strings"
)

// ConnectionCheckType identifies which phase of the connection test failed.
type ConnectionCheckType string

const (
	// ConnectionCheckNetwork indicates a TCP/DNS reachability failure.
	ConnectionCheckNetwork ConnectionCheckType = "network"
	// ConnectionCheckAuth indicates a credential rejection (invalid key, bad password, etc.).
	ConnectionCheckAuth ConnectionCheckType = "auth"
	// ConnectionCheckAccess indicates valid credentials but insufficient permissions or a missing resource.
	ConnectionCheckAccess ConnectionCheckType = "access"
)

// ConnectionCheckError is returned when one of the three sequential connection checks fails.
// It carries the check type and a human-readable message so callers can surface a specific,
// actionable error to the user.
type ConnectionCheckError struct {
	// CheckType is the phase of the test that failed (network, auth, or access).
	CheckType ConnectionCheckType
	// Message describes the failure in human-readable terms.
	Message string
}

func (e *ConnectionCheckError) Error() string {
	return fmt.Sprintf("[%s] %s", strings.ToUpper(string(e.CheckType)), e.Message)
}

// ConnectionTester is the interface implemented by each datasource provider.
// It captures only the behaviour that differs between providers; the generic
// CreateDatasource flow (validation, encryption, DB insert) lives in DatasourceService
// and delegates to this interface for the parts that vary.
type ConnectionTester interface {
	// TestConnection runs provider-specific connectivity checks (network → auth → access).
	// Returns nil when all checks pass, or a *ConnectionCheckError on the first failure.
	TestConnection(ctx context.Context, params map[string]any) error
}

// DataResult wraps the retrieved data payload and optional metadata.
type DataResult struct {
	Metadata map[string]any `json:"metadata,omitempty"`
	Data     any            `json:"data"`
}

// DatasourceDataProvider is an optional interface implemented by datasource providers
// that support live data or schema querying.
type DatasourceDataProvider interface {
	// ValidateDataRequest performs provider-level validation for query parameters.
	// Returns a *ValidationError (e.g. 400 Bad Request) if parameters are missing or invalid.
	ValidateDataRequest(ctx context.Context, queryParams map[string]string) error

	// FetchData executes the query against the datasource using decrypted connection params.
	FetchData(ctx context.Context, params map[string]any, queryParams map[string]string) (*DataResult, error)
}

// Made with Bob
