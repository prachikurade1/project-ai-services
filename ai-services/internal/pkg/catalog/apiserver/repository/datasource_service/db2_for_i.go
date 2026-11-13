package datasourceservice

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	mapepire "github.com/deady54/mapepire-go"
)

const (
	// db2ForIDefaultPort is the default Mapepire daemon port.
	db2ForIDefaultPort = "8076"
	// db2ForITimeout bounds each phase (network, auth, access) of the connection test.
	db2ForITimeout = 10 * time.Second
	// db2ForIJobID is the identifier of the short-lived Mapepire SQL job used for the test.
	db2ForIJobID = "db2-for-i-test"
	// db2ForIAccessQuery is a read-only probe that returns the relational database (RDB) name
	// the job is connected to.
	db2ForIAccessQuery = "SELECT CURRENT SERVER AS DB_NAME FROM SYSIBM.SYSDUMMY1"
	// db2ForIProperties works around a misspelled JSON key in the mapepire-go SDK connect payload: the value injects
	// "technique":"TCP" into the connect payload, forcing the TCP Db2 driver and avoiding the
	// CCSID 65535 exception. It must be set verbatim.
	db2ForIProperties = "translate binary=true;ccsid=37\",\"technique\":\"TCP"
)

// db2ForITester implements ConnectionTester for IBM Db2 for i accessed through Mapepire.
type db2ForITester struct{}

// NewDb2ForITester returns a ConnectionTester for IBM Db2 for i providers.
func NewDb2ForITester() ConnectionTester {
	return &db2ForITester{}
}

// TestConnection runs three sequential checks against a Mapepire daemon:
//  1. Network — TCP dial to host:port (port defaults to 8076 when not supplied).
//  2. Auth    — Mapepire WebSocket handshake and IBM i authentication.
//  3. Access  — read-only query confirming the session is connected to the configured database.
//
// TLS certificates are always verified.
func (t *db2ForITester) TestConnection(ctx context.Context, params map[string]any) error {
	host, _ := params["host"].(string)
	database, _ := params["database"].(string)
	username, _ := params["username"].(string)
	password, _ := params["password"].(string)
	port := db2ForIPort(params["port"])

	if err := checkDb2ForINetwork(ctx, net.JoinHostPort(host, port)); err != nil {
		return err
	}

	job, err := connectDb2ForIJob(ctx, mapepire.DaemonServer{
		Host:       host,
		User:       username,
		Password:   password,
		Port:       port,
		Properties: db2ForIProperties,
	}, username)
	if err != nil {
		return err
	}
	defer func() { _ = job.Close() }()

	return checkDb2ForIAccess(ctx, job, database)
}

// db2ForIPort resolves the port from the stored parameter, which may be a string, a JSON
// number or an integer. It falls back to the Mapepire default when absent or empty.
func db2ForIPort(raw any) string {
	switch v := raw.(type) {
	case string:
		if v != "" {
			return v
		}
	case float64:
		return strconv.Itoa(int(v))
	case int:
		return strconv.Itoa(v)
	}

	return db2ForIDefaultPort
}

// checkDb2ForINetwork verifies TCP reachability of the Mapepire daemon.
func checkDb2ForINetwork(ctx context.Context, addr string) error {
	dialCtx, cancel := context.WithTimeout(ctx, db2ForITimeout)
	defer cancel()

	dialer := &net.Dialer{Timeout: db2ForITimeout}

	conn, err := dialer.DialContext(dialCtx, "tcp", addr)
	if err != nil {
		return &ConnectionCheckError{
			CheckType: ConnectionCheckNetwork,
			Message:   fmt.Sprintf("TCP dial %s failed: %v", addr, err),
		}
	}
	// The Mapepire SDK opens its own WebSocket connection.
	_ = conn.Close()

	return nil
}

// connectDb2ForIJob opens an authenticated Mapepire SQL job. Connect does not accept a
// context, so it is raced against the phase timeout; a job that connects late is closed.
func connectDb2ForIJob(ctx context.Context, creds mapepire.DaemonServer, username string) (*mapepire.SQLJob, error) {
	job := mapepire.NewSQLJob(db2ForIJobID)

	_, err := runDb2ForIStep(ctx, func() (struct{}, error) {
		return struct{}{}, job.Connect(creds)
	}, func(struct{}) { _ = job.Close() })
	if err != nil {
		return nil, &ConnectionCheckError{
			CheckType: ConnectionCheckAuth,
			Message:   fmt.Sprintf("Database authentication failed for user %s: check username and password", username),
		}
	}

	return job, nil
}

// checkDb2ForIAccess confirms the authenticated user can run a read-only query and that the
// session is connected to the configured database (compared case-insensitively against
// CURRENT SERVER, the relational database name of the system).
func checkDb2ForIAccess(ctx context.Context, job *mapepire.SQLJob, database string) error {
	res, err := runDb2ForIStep(ctx, func() (*mapepire.ServerResponse, error) {
		q, err := job.Query(db2ForIAccessQuery)
		if err != nil {
			return nil, err
		}

		return q.Execute()
	}, nil)
	if err == nil && (res == nil || !res.Success) {
		err = errors.New("query reported success = false")
	}
	if err != nil {
		return &ConnectionCheckError{
			CheckType: ConnectionCheckAccess,
			Message:   fmt.Sprintf("Database query test failed: %v", err),
		}
	}

	actual := db2ForIFirstValue(res.Data)
	if !strings.EqualFold(strings.TrimSpace(actual), strings.TrimSpace(database)) {
		return &ConnectionCheckError{
			CheckType: ConnectionCheckAccess,
			Message:   fmt.Sprintf("Database %q is not accessible: the system reports database %q", database, actual),
		}
	}

	return nil
}

// db2ForIFirstValue returns the first column value of the first row as a string, or an
// empty string when the result is empty.
func db2ForIFirstValue(rows []map[string]any) string {
	if len(rows) == 0 {
		return ""
	}

	for _, v := range rows[0] {
		return fmt.Sprint(v)
	}

	return ""
}

// runDb2ForIStep runs fn in a goroutine and waits for it or for the phase timeout. If the
// step finishes after the timeout, cleanup (when non-nil) is invoked with its result.
func runDb2ForIStep[T any](ctx context.Context, fn func() (T, error), cleanup func(T)) (T, error) {
	type outcome struct {
		val T
		err error
	}

	stepCtx, cancel := context.WithTimeout(ctx, db2ForITimeout)
	defer cancel()

	done := make(chan outcome, 1)

	go func() {
		val, err := fn()
		done <- outcome{val: val, err: err}
	}()

	select {
	case o := <-done:
		return o.val, o.err
	case <-stepCtx.Done():
		if cleanup != nil {
			go func() {
				if o := <-done; o.err == nil {
					cleanup(o.val)
				}
			}()
		}

		var zero T

		return zero, fmt.Errorf("timed out after %v: %w", db2ForITimeout, stepCtx.Err())
	}
}
