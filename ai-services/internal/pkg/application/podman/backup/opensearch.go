package backup

import (
	"context"
	"fmt"
	"strings"
	"time"

	commonBackup "github.com/project-ai-services/ai-services/internal/pkg/application/common/backup"
	"github.com/project-ai-services/ai-services/internal/pkg/logger"
	"github.com/project-ai-services/ai-services/internal/pkg/runtime"
	"github.com/project-ai-services/ai-services/internal/pkg/vars"
)

// BackupOpenSearch performs OpenSearch backup using a sidecar container.
// rt is the runtime.Runtime for the worker that hosts the OpenSearch pod — it
// may be a local PodmanClient or a RemoteRuntime that forwards calls over gRPC.
// podName is the OpenSearch pod name; it is accepted by both CreateSidecarContainer
// (specgen.Pod field) and InspectPod.
func BackupOpenSearch(ctx context.Context, rt runtime.Runtime, podName, backupFile string) error {
	sidecarName := fmt.Sprintf("opensearch-backup-sidecar-%d", time.Now().Unix())

	containerID, err := rt.CreateSidecarContainer(ctx, podName, sidecarName, vars.ToolImage, []string{"sleep", "3600"})
	if err != nil {
		return fmt.Errorf("failed to create backup sidecar: %w", err)
	}

	defer func() {
		logger.Infoln("Cleaning up backup sidecar container...")
		if stopErr := rt.StopContainer(ctx, containerID); stopErr != nil {
			logger.Warningf("Failed to stop backup sidecar %s: %v\n", containerID, stopErr)
		}
		logger.Infoln("Backup sidecar cleanup completed")
	}()

	return prepareSidecarAndBackup(ctx, rt, podName, containerID, backupFile)
}

// prepareSidecarAndBackup prepares the sidecar container and performs the backup.
// podName is the OpenSearch pod name used for InspectPod; containerID is the sidecar.
func prepareSidecarAndBackup(ctx context.Context, rt runtime.Runtime, podName, containerID, backupFile string) error {
	// Get OpenSearch password from the pod's mounted secret via InspectPod
	osPassword, err := getOpenSearchPasswordFromSecret(ctx, rt, podName)
	if err != nil {
		return fmt.Errorf("failed to get OpenSearch password: %w", err)
	}

	// Create backup directory in container
	containerBackupPath := "/tmp/opensearch_backup"
	if _, err := rt.ExecInContainerWithCmd(ctx, containerID, "", []string{"mkdir", "-p", containerBackupPath}); err != nil {
		return fmt.Errorf("failed to create backup directory in container: %w", err)
	}

	// Perform backup using curl
	if err := performBackupWithCurl(ctx, rt, containerID, "localhost:9200", osPassword, containerBackupPath); err != nil {
		return fmt.Errorf("backup failed: %w", err)
	}

	// Copy backup files from container to host, then create tar archive on host
	if err := CopyAndTarBackup(ctx, rt, containerID, containerBackupPath, backupFile); err != nil {
		return fmt.Errorf("failed to copy and archive backup: %w", err)
	}

	logger.Infoln("OpenSearch backup completed!")

	return nil
}

// performBackupWithCurl performs the OpenSearch backup using curl commands in container.
func performBackupWithCurl(ctx context.Context, rt runtime.Runtime, containerID, osHost, osPassword, backupDir string) error {
	logger.Infoln("Exporting OpenSearch indices...")

	indices, err := listRagIndices(ctx, rt, containerID, osHost, osPassword)
	if err != nil {
		return err
	}

	if len(indices) == 0 {
		logger.Warningf("No indices found starting with 'rag'\n")

		return nil
	}

	logger.Infof("Found %d indices to backup\n", len(indices))

	backedUpCount, lastErr := backupIndices(ctx, rt, containerID, osHost, osPassword, backupDir, indices)

	if err := handleBackupResults(backedUpCount, len(indices), lastErr); err != nil {
		return err
	}

	// Create backup_info.json
	if err := createBackupInfo(ctx, rt, containerID, backupDir); err != nil {
		logger.Warningf("Failed to create backup_info.json: %v\n", err)
	}

	return nil
}

// listRagIndices lists all indices that start with "rag".
func listRagIndices(ctx context.Context, rt runtime.Runtime, containerID, osHost, osPassword string) ([]string, error) {
	listScript := commonBackup.ListRagIndicesScript(osHost)
	wrappedScript := wrapScriptWithPassword(osPassword, listScript)

	output, err := rt.ExecInContainerWithCmd(ctx, containerID, "", []string{"sh", "-c", wrappedScript})
	if err != nil {
		return nil, fmt.Errorf("failed to list indices: %w, output: %s", err, output)
	}

	return commonBackup.ParseIndicesList(output), nil
}

// backupIndices backs up all provided indices and returns the count and any error.
func backupIndices(ctx context.Context, rt runtime.Runtime, containerID, osHost, osPassword, backupDir string, indices []string) (int, error) {
	backedUpCount := 0
	var lastErr error

	for _, indexName := range indices {
		if err := commonBackup.CheckContextCancellation(ctx, backedUpCount); err != nil {
			return backedUpCount, err
		}

		indexName = strings.TrimSpace(indexName)
		if indexName == "" {
			continue
		}

		if err := backupIndexWithCurl(ctx, rt, containerID, osHost, osPassword, backupDir, indexName); err != nil {
			logger.Errorf("Failed to backup index %s: %v\n", indexName, err)
			lastErr = err

			continue
		}

		backedUpCount++
	}

	return backedUpCount, lastErr
}

// handleBackupResults checks backup results and logs appropriate messages.
func handleBackupResults(backedUpCount, totalCount int, lastErr error) error {
	return commonBackup.HandleBackupResults(backedUpCount, totalCount, lastErr)
}

// backupIndexWithCurl backs up a single index using curl in container.
func backupIndexWithCurl(ctx context.Context, rt runtime.Runtime, containerID, osHost, osPassword, backupDir, indexName string) error {
	logger.Infof("  Exporting index: %s\n", indexName)

	if err := exportIndexMetadata(ctx, rt, containerID, osHost, osPassword, backupDir, indexName); err != nil {
		return err
	}

	if err := exportIndexData(ctx, rt, containerID, osHost, osPassword, backupDir, indexName); err != nil {
		return err
	}

	countDocuments(ctx, rt, containerID, backupDir, indexName)

	return nil
}

// exportIndexMetadata exports mapping and settings for an index.
func exportIndexMetadata(ctx context.Context, rt runtime.Runtime, containerID, osHost, osPassword, backupDir, indexName string) error {
	// Export mapping
	mappingScript := commonBackup.GenerateExportMappingScript(osHost, indexName, backupDir)
	wrappedMapping := wrapScriptWithPassword(osPassword, mappingScript)
	if _, err := rt.ExecInContainerWithCmd(ctx, containerID, "", []string{"sh", "-c", wrappedMapping}); err != nil {
		return fmt.Errorf("failed to export mapping: %w", err)
	}

	// Export settings
	settingsScript := commonBackup.GenerateExportSettingsScript(osHost, indexName, backupDir)
	wrappedSettings := wrapScriptWithPassword(osPassword, settingsScript)
	if _, err := rt.ExecInContainerWithCmd(ctx, containerID, "", []string{"sh", "-c", wrappedSettings}); err != nil {
		return fmt.Errorf("failed to export settings: %w", err)
	}

	return nil
}

// exportIndexData exports all documents from an index using scroll API.
func exportIndexData(ctx context.Context, rt runtime.Runtime, containerID, osHost, osPassword, backupDir, indexName string) error {
	// First, initiate scroll
	scrollInitScript := commonBackup.GenerateScrollInitScript(osHost, indexName)
	wrappedInit := wrapScriptWithPassword(osPassword, scrollInitScript)
	if _, err := rt.ExecInContainerWithCmd(ctx, containerID, "", []string{"sh", "-c", wrappedInit}); err != nil {
		return fmt.Errorf("failed to initiate scroll: %w", err)
	}

	// Extract scroll_id and hits with improved error handling and loop protection
	extractScript := commonBackup.GenerateScrollExportScript(osHost, backupDir, indexName)
	wrappedExtract := wrapScriptWithPassword(osPassword, extractScript)
	if _, err := rt.ExecInContainerWithCmd(ctx, containerID, "", []string{"sh", "-c", wrappedExtract}); err != nil {
		return fmt.Errorf("failed to export data: %w", err)
	}

	return nil
}

// countDocuments counts and logs the number of documents in the backup.
func countDocuments(ctx context.Context, rt runtime.Runtime, containerID, backupDir, indexName string) {
	countScript := commonBackup.GenerateCountDocumentsScript(backupDir, indexName)
	countOutput, err := rt.ExecInContainerWithCmd(ctx, containerID, "", []string{"sh", "-c", countScript})
	commonBackup.LogDocumentCount(countOutput, err)
}

// createBackupInfo creates a backup_info.json file with metadata.
func createBackupInfo(ctx context.Context, rt runtime.Runtime, containerID, backupDir string) error {
	infoScript := commonBackup.GenerateBackupInfoScript(backupDir)
	_, err := rt.ExecInContainerWithCmd(ctx, containerID, "", []string{"sh", "-c", infoScript})

	return err
}

// wrapScriptWithPassword wraps a script with password environment variable setup.
func wrapScriptWithPassword(password, script string) string {
	escapedPassword := strings.ReplaceAll(password, "'", "'\\''")

	return fmt.Sprintf(`
OS_PASSWORD='%s'
export OS_PASSWORD
%s
`, escapedPassword, script)
}

// getOpenSearchPasswordFromSecret retrieves the OpenSearch admin password by
// reading the secret file mounted inside the sidecar container. The sidecar
// shares the pod's secret mounts at /run/secrets/<secret-name>, where
// <secret-name> is stored in the pod label ai-services.io/secret. We get the
// secret name via InspectPod (works for both local and remote runtimes), then
// exec a cat inside the sidecar to read the file contents.
func getOpenSearchPasswordFromSecret(ctx context.Context, rt runtime.Runtime, podID string) (string, error) {
	pod, err := rt.InspectPod(ctx, podID)
	if err != nil {
		return "", fmt.Errorf("failed to inspect pod: %w", err)
	}

	secretName, ok := pod.Labels["ai-services.io/secret"]
	if !ok || secretName == "" {
		return "", fmt.Errorf("secret label 'ai-services.io/secret' not found in pod labels")
	}

	logger.Infof("Reading password from secret: %s\n", secretName)

	secretPath := fmt.Sprintf("/run/secrets/%s", secretName)
	output, err := rt.ExecInContainerWithCmd(ctx, podID, "", []string{"cat", secretPath})
	if err != nil {
		return "", fmt.Errorf("failed to read secret file %s: %w", secretPath, err)
	}

	password, err := extractPasswordFromSecretData(output)
	if err != nil {
		return "", err
	}

	logger.Infoln("Successfully retrieved password from secret")

	return password, nil
}

// extractPasswordFromSecretData parses key:value secret data to find the password field.
func extractPasswordFromSecretData(secretData string) (string, error) {
	const keyValueParts = 2

	for _, line := range strings.Split(secretData, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		parts := strings.SplitN(line, ":", keyValueParts)
		if len(parts) != keyValueParts {
			continue
		}

		if strings.TrimSpace(parts[0]) == "password" {
			if v := strings.TrimSpace(parts[1]); v != "" {
				return v, nil
			}
		}
	}

	return "", fmt.Errorf("password field not found in secret data")
}
