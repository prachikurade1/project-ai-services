package datasourceservice

import (
	"context"
	"net/http"
	"slices"
	"testing"

	"github.com/google/uuid"
	"github.com/project-ai-services/ai-services/internal/pkg/catalog"
	apimodels "github.com/project-ai-services/ai-services/internal/pkg/catalog/apiserver/models"
	catalogconstants "github.com/project-ai-services/ai-services/internal/pkg/catalog/constants"
	dbmodels "github.com/project-ai-services/ai-services/internal/pkg/catalog/db/models"
	dbrepo "github.com/project-ai-services/ai-services/internal/pkg/catalog/db/repository"
	"github.com/project-ai-services/ai-services/internal/pkg/catalog/validators"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockAppRepoForTest struct {
	app *dbmodels.Application
	err error
}

func (m *mockAppRepoForTest) GetAll(_ context.Context, _ *dbrepo.ApplicationFilters) ([]dbmodels.Application, error) {
	return nil, nil
}
func (m *mockAppRepoForTest) GetCount(_ context.Context, _ *dbrepo.ApplicationFilters) (int, error) {
	return 0, nil
}
func (m *mockAppRepoForTest) GetByID(_ context.Context, _ uuid.UUID) (*dbmodels.Application, error) {
	return m.app, m.err
}
func (m *mockAppRepoForTest) GetByName(_ context.Context, _ string) (*dbmodels.Application, error) {
	return nil, nil
}
func (m *mockAppRepoForTest) Insert(_ context.Context, _ *dbmodels.Application) error {
	return nil
}
func (m *mockAppRepoForTest) UpdateDeploymentName(_ context.Context, _ uuid.UUID, _ string) error {
	return nil
}
func (m *mockAppRepoForTest) UpdateStatus(_ context.Context, _ uuid.UUID, _ dbmodels.ApplicationStatus, _ string) error {
	return nil
}
func (m *mockAppRepoForTest) Delete(_ context.Context, _ uuid.UUID) error {
	return nil
}

func TestConnectDatasourcesToApplication_OwnershipCheck(t *testing.T) {
	appID := uuid.New()
	dsID := uuid.New()

	app := &dbmodels.Application{
		ID:        appID,
		Name:      "test-app",
		CreatedBy: "victim-user",
	}

	appRepo := &mockAppRepoForTest{app: app}
	svc := &DatasourceService{
		appRepo: appRepo,
	}

	// 1. Attacker calls connect on victim-owned application -> 403 Forbidden
	resp, err := svc.ConnectDatasourcesToApplication(context.Background(), appID, "attacker-user", []uuid.UUID{dsID})
	assert.Nil(t, resp)
	require.Error(t, err)

	valErr, ok := err.(*ValidationError)
	require.True(t, ok)
	assert.Equal(t, http.StatusForbidden, valErr.Code)
	assert.Equal(t, "user does not own this application", valErr.Message)
}

func TestDisconnectDatasourcesFromApplication_OwnershipCheck(t *testing.T) {
	appID := uuid.New()
	dsID := uuid.New()

	app := &dbmodels.Application{
		ID:        appID,
		Name:      "test-app",
		CreatedBy: "victim-user",
	}

	appRepo := &mockAppRepoForTest{app: app}
	svc := &DatasourceService{
		appRepo: appRepo,
	}

	// 1. Attacker calls disconnect on victim-owned application -> 403 Forbidden
	err := svc.DisconnectDatasourcesFromApplication(context.Background(), appID, "attacker-user", dsID)
	require.Error(t, err)

	valErr, ok := err.(*ValidationError)
	require.True(t, ok)
	assert.Equal(t, http.StatusForbidden, valErr.Code)
	assert.Equal(t, "user does not own this application", valErr.Message)
}

// mockConnectorRepoForTest embeds the interface so only GetByIDs needs to be implemented.
type mockConnectorRepoForTest struct {
	dbrepo.ConnectorRepository
}

func (m *mockConnectorRepoForTest) GetByIDs(_ context.Context, _ []uuid.UUID) (map[uuid.UUID]dbmodels.Connector, error) {
	return nil, nil
}

func TestBuildDatasourcePage_PreservesOrder(t *testing.T) {
	const n = 5

	order := make([]string, n)
	for i := range order {
		order[i] = uuid.New().String()
	}

	// Insert in the reverse of the expected order. Go map iteration for small maps is a
	// rotation of insertion order, so ranging over the map can never reproduce `order`.
	serviceItems := make(map[string]apimodels.ConnectorItem, n)
	for i := n - 1; i >= 0; i-- {
		serviceItems[order[i]] = apimodels.ConnectorItem{ID: order[i]}
	}

	svc := &DatasourceService{connectorRepo: &mockConnectorRepoForTest{}}

	items, err := svc.buildDatasourcePage(context.Background(), "http://service", serviceItems, order)
	require.NoError(t, err)
	require.Len(t, items, n)

	for i, item := range items {
		assert.Equal(t, order[i], item.ID)
	}
}

// mockSvcDepRepoForCount embeds the interface so only GetDependenciesByServiceID needs to be implemented.
type mockSvcDepRepoForCount struct {
	dbrepo.ServiceDependencyRepository
	deps []dbmodels.ServiceDependency
}

func (m *mockSvcDepRepoForCount) GetDependenciesByServiceID(_ context.Context, _ uuid.UUID) ([]dbmodels.ServiceDependency, error) {
	return m.deps, nil
}

// mockConnectorRepoByIDs embeds the interface so only GetByIDs needs to be implemented.
type mockConnectorRepoByIDs struct {
	dbrepo.ConnectorRepository
	connectors map[uuid.UUID]dbmodels.Connector
}

func (m *mockConnectorRepoByIDs) GetByIDs(_ context.Context, _ []uuid.UUID) (map[uuid.UUID]dbmodels.Connector, error) {
	return m.connectors, nil
}

func TestCountDatasourcesForService_MixedConnectorTypes(t *testing.T) {
	dsID1, dsID2, modelID := uuid.New(), uuid.New(), uuid.New()

	svc := &DatasourceService{
		svcDepRepo: &mockSvcDepRepoForCount{deps: []dbmodels.ServiceDependency{
			{DependencyID: dsID1, DependencyType: dbmodels.DependencyTypeConnector},
			{DependencyID: dsID2, DependencyType: dbmodels.DependencyTypeConnector},
			{DependencyID: modelID, DependencyType: dbmodels.DependencyTypeConnector},
			{DependencyID: uuid.New(), DependencyType: dbmodels.DependencyTypeComponent},
		}},
		connectorRepo: &mockConnectorRepoByIDs{connectors: map[uuid.UUID]dbmodels.Connector{
			dsID1:   {ID: dsID1, Type: catalogconstants.ConnectorTypeDatasource},
			dsID2:   {ID: dsID2, Type: catalogconstants.ConnectorTypeDatasource},
			modelID: {ID: modelID, Type: catalogconstants.ConnectorTypeModels},
		}},
	}

	count, err := svc.countDatasourcesForService(context.Background(), uuid.New())
	require.NoError(t, err)
	assert.Equal(t, 2, count, "model connectors and non-connector dependencies must not count")
}

func TestCountDatasourcesForService_NoConnectors(t *testing.T) {
	svc := &DatasourceService{
		svcDepRepo: &mockSvcDepRepoForCount{deps: []dbmodels.ServiceDependency{
			{DependencyID: uuid.New(), DependencyType: dbmodels.DependencyTypeComponent},
		}},
	}

	count, err := svc.countDatasourcesForService(context.Background(), uuid.New())
	require.NoError(t, err)
	assert.Equal(t, 0, count)
}

func TestCountDatasourceConnectors_MixedTypes(t *testing.T) {
	identity := func(s string) string { return s }

	got := validators.CountDatasourceConnectors(slices.Values([]string{
		catalogconstants.ConnectorTypeDatasource,
		catalogconstants.ConnectorTypeModels,
		catalogconstants.ConnectorTypeDatasource,
		"other",
	}), identity)

	assert.Equal(t, 2, got)
	assert.Equal(t, 0, validators.CountDatasourceConnectors(slices.Values([]string(nil)), identity))
}

type mockConnectorRepoForData struct {
	dbrepo.ConnectorRepository
	conn *dbmodels.Connector
	err  error
}

func (m *mockConnectorRepoForData) GetByID(_ context.Context, _ uuid.UUID, _ bool) (*dbmodels.Connector, error) {
	return m.conn, m.err
}

type mockDataProviderForTest struct {
	validateErr error
	fetchResult *DataResult
	fetchErr    error
}

func (m *mockDataProviderForTest) ValidateDataRequest(_ context.Context, _ map[string]string) error {
	return m.validateErr
}

func (m *mockDataProviderForTest) FetchData(_ context.Context, _ map[string]any, _ map[string]string) (*DataResult, error) {
	return m.fetchResult, m.fetchErr
}

func TestGetDatasourceData_NotFound(t *testing.T) {
	svc := &DatasourceService{
		connectorRepo: &mockConnectorRepoForData{conn: nil},
	}

	resp, err := svc.GetDatasourceData(context.Background(), uuid.New(), map[string]string{"type": "tables"})
	assert.Nil(t, resp)
	require.Error(t, err)

	valErr, ok := err.(*ValidationError)
	require.True(t, ok)
	assert.Equal(t, http.StatusNotFound, valErr.Code)
}

func TestGetDatasourceData_UnsupportedProvider(t *testing.T) {
	dsID := uuid.New()
	conn := &dbmodels.Connector{
		ID:       dsID,
		Provider: catalogconstants.DatasourceProviderObjectStorage,
	}

	svc := &DatasourceService{
		connectorRepo: &mockConnectorRepoForData{conn: conn},
		dataProviders: map[string]DatasourceDataProvider{},
	}

	resp, err := svc.GetDatasourceData(context.Background(), dsID, map[string]string{"type": "tables"})
	assert.Nil(t, resp)
	require.Error(t, err)

	valErr, ok := err.(*ValidationError)
	require.True(t, ok)
	assert.Equal(t, http.StatusBadRequest, valErr.Code)
	assert.Contains(t, valErr.Message, "does not support the data query API")
}

func TestGetDatasourceData_ValidationFailure(t *testing.T) {
	dsID := uuid.New()
	conn := &dbmodels.Connector{
		ID:       dsID,
		Provider: catalogconstants.DatasourceProviderDb2ForI,
	}

	svc := &DatasourceService{
		connectorRepo: &mockConnectorRepoForData{conn: conn},
		dataProviders: map[string]DatasourceDataProvider{
			catalogconstants.DatasourceProviderDb2ForI: &mockDataProviderForTest{
				validateErr: &ValidationError{Code: http.StatusBadRequest, Message: "Query parameter 'type' is required"},
			},
		},
	}

	resp, err := svc.GetDatasourceData(context.Background(), dsID, map[string]string{})
	assert.Nil(t, resp)
	require.Error(t, err)

	valErr, ok := err.(*ValidationError)
	require.True(t, ok)
	assert.Equal(t, http.StatusBadRequest, valErr.Code)
	assert.Equal(t, "Query parameter 'type' is required", valErr.Message)
}

func TestGetDatasourceData_Success(t *testing.T) {
	dsID := uuid.New()
	conn := &dbmodels.Connector{
		ID:       dsID,
		Provider: catalogconstants.DatasourceProviderDb2ForI,
		Metadata: map[string]any{},
	}

	catProvider, err := catalog.NewCatalogProvider(nil)
	require.NoError(t, err)

	expectedTables := []string{"ORDERS", "CUSTOMERS"}
	expectedMetadata := map[string]any{"schema": "PRODDB", "total": 2}

	svc := &DatasourceService{
		connectorRepo:   &mockConnectorRepoForData{conn: conn},
		catalogProvider: catProvider,
		encryptionKey:   "01234567890123456789012345678901",
		dataProviders: map[string]DatasourceDataProvider{
			catalogconstants.DatasourceProviderDb2ForI: &mockDataProviderForTest{
				fetchResult: &DataResult{
					Metadata: expectedMetadata,
					Data:     expectedTables,
				},
			},
		},
	}

	resp, err := svc.GetDatasourceData(context.Background(), dsID, map[string]string{"type": "tables"})
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.Equal(t, dsID, resp.DatasourceID)
	assert.Equal(t, catalogconstants.DatasourceProviderDb2ForI, resp.Provider)
	assert.Equal(t, "tables", resp.Type)
	assert.Equal(t, expectedMetadata, resp.Metadata)
	assert.Equal(t, expectedTables, resp.Data)
}

func TestDb2ForIDataProvider_ValidateDataRequest(t *testing.T) {
	provider := NewDb2ForIDataProvider()

	// Missing type.
	err := provider.ValidateDataRequest(context.Background(), map[string]string{})
	require.Error(t, err)
	valErr, ok := err.(*ValidationError)
	require.True(t, ok)
	assert.Equal(t, http.StatusBadRequest, valErr.Code)
	assert.Equal(t, "Query parameter 'type' is required", valErr.Message)

	// Invalid type.
	err = provider.ValidateDataRequest(context.Background(), map[string]string{"type": "indexes"})
	require.Error(t, err)
	valErr, ok = err.(*ValidationError)
	require.True(t, ok)
	assert.Equal(t, http.StatusBadRequest, valErr.Code)
	assert.Contains(t, valErr.Message, "supported types for 'db2_for_i' are [tables, columns]")

	// Valid tables.
	err = provider.ValidateDataRequest(context.Background(), map[string]string{"type": "tables"})
	require.NoError(t, err)

	// Columns missing table.
	err = provider.ValidateDataRequest(context.Background(), map[string]string{"type": "columns"})
	require.Error(t, err)
	valErr, ok = err.(*ValidationError)
	require.True(t, ok)
	assert.Equal(t, http.StatusBadRequest, valErr.Code)
	assert.Equal(t, "Parameter 'table' is required when type is 'columns'", valErr.Message)

	// Columns with table.
	err = provider.ValidateDataRequest(context.Background(), map[string]string{"type": "columns", "table": "ORDERS"})
	require.NoError(t, err)
}

func TestExtractString(t *testing.T) {
	row := map[string]any{
		"TABLE_NAME": "ORDERS",
		"NUM_VAL":    123,
	}

	assert.Equal(t, "ORDERS", extractString(row, "TABLE_NAME"))
	assert.Equal(t, "ORDERS", extractString(row, "table_name"))
	assert.Equal(t, "123", extractString(row, "num_val"))
	assert.Equal(t, "", extractString(row, "NON_EXISTENT"))
}
