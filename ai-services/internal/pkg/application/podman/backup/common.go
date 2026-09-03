package backup

import (
	"context"
	"fmt"
	"os"

	"github.com/project-ai-services/ai-services/internal/pkg/logger"
	"github.com/project-ai-services/ai-services/internal/pkg/runtime"
)

// CopyAndTarBackup streams the backup directory and the backup_info.json file
// out of the sidecar container using CopyFromContainer (which runs `podman cp
// <container>:<path> -` on the worker) and writes the resulting tar archive to
// backupFile on the local machine.
//
// This approach works for both a local PodmanClient and a RemoteRuntime because
// CopyFromContainer is on the runtime.Runtime interface — no direct `podman cp`
// invocation is needed on the control-plane side.
func CopyAndTarBackup(ctx context.Context, rt runtime.Runtime, containerID, containerBackupPath, backupFile string) error {
	logger.Infof("Streaming backup archive from container to host...\n")

	// podman cp produces a POSIX tar archive. Passing "-" as the destination
	// writes the archive to stdout. The tar contains both the backup_info.json
	// at /tmp/backup_info.json and the opensearch_backup directory under
	// containerBackupPath — we copy each into separate archives and then merge,
	// OR we copy the whole /tmp directory root that contains both.
	//
	// Simplest: copy /tmp/backup_info.json and containerBackupPath separately,
	// then concatenate into one tar.gz on the host using the existing archive helper.

	// Stream containerBackupPath (opensearch_backup/) out of the container.
	backupDirBytes, err := rt.CopyFromContainer(ctx, containerID, containerBackupPath)
	if err != nil {
		return fmt.Errorf("failed to copy backup directory from container: %w", err)
	}

	// Stream backup_info.json out of the container.
	backupInfoBytes, err := rt.CopyFromContainer(ctx, containerID, "/tmp/backup_info.json")
	if err != nil {
		return fmt.Errorf("failed to copy backup_info.json from container: %w", err)
	}

	// Write a combined archive.
	// podman cp produces a tar archive per call; we create one merged tar.gz on
	// the host from both tar streams using the shared archive helpers.
	if err := writeCombinedArchive(backupDirBytes, backupInfoBytes, backupFile); err != nil {
		return fmt.Errorf("failed to write backup archive: %w", err)
	}

	logger.Infof("✓ Backup files copied to host\n")
	LogArchiveSize(backupFile)

	return nil
}

// writeCombinedArchive merges two raw tar streams (from podman cp) into a
// single .tar.gz file at dest.
func writeCombinedArchive(backupDirTar, backupInfoTar []byte, dest string) error {
	tempDir, err := os.MkdirTemp("", "opensearch-backup-*")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}

	defer func() {
		if removeErr := os.RemoveAll(tempDir); removeErr != nil {
			logger.Warningf("Failed to remove temp directory: %v\n", removeErr)
		}
	}()

	// Extract both tar streams into tempDir, then call CreateTarGzArchive.
	if err := ExtractTarBytes(backupInfoTar, tempDir); err != nil {
		return fmt.Errorf("failed to extract backup_info.json: %w", err)
	}

	backupSubDir := tempDir + "/opensearch_backup"

	const dirPerm = 0o755
	if err := os.MkdirAll(backupSubDir, dirPerm); err != nil {
		return fmt.Errorf("failed to create opensearch_backup dir: %w", err)
	}

	if err := ExtractTarBytes(backupDirTar, backupSubDir); err != nil {
		return fmt.Errorf("failed to extract opensearch_backup: %w", err)
	}

	return CreateTarGzArchive(tempDir, dest, []string{"backup_info.json", "opensearch_backup"})
}
