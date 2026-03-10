## Flowchart: `SauterQuadratureTaskRingBuffer::add_task()`

```mermaid
flowchart TD
  Start([add_task])
  Lock1[Acquire buffer lock unique_lock]
  WaitSpace{Ring buffer has space}
  Reserve[Reserve task index and advance tail_pending]
  Unlock1[Release buffer lock]

  SetDuring[Set status during_creation]
  Fill[Fill task payload in host ring buffers]
  SetCreated[Set status created]

  Lock2[Acquire buffer lock lock_guard]
  MoveTail{task index equals tail_committed}
  AdvanceTail[Advance tail_committed while next is created]
  DecideNotify[Compute ready task count]
  NotifyRule{Notify consumers now}
  Unlock2[Release buffer lock]
  Notify[Notify all consumers]
  End([return])

  Start --> Lock1
  Lock1 --> WaitSpace
  WaitSpace -->|no| WaitSpace
  WaitSpace -->|yes| Reserve
  Reserve --> Unlock1

  Unlock1 --> SetDuring
  SetDuring --> Fill
  Fill --> SetCreated

  SetCreated --> Lock2
  Lock2 --> MoveTail
  MoveTail -->|yes| AdvanceTail
  MoveTail -->|no| DecideNotify
  AdvanceTail --> DecideNotify
  DecideNotify --> NotifyRule
  NotifyRule --> Unlock2

  Unlock2 -->|yes| Notify
  Unlock2 -->|no| End
  Notify --> End
```

### Notes
- **Ring-buffer pointers**
  - `tail_pending`: next insertion position reserved for a producer (advanced under `buffer_lock` before payload is written).
  - `tail_committed`: end of the contiguous “ready-for-processing” region; advanced only when the current task fills the next gap.
- **Statuses**
  - `during_creation` is set before writing payload fields.
  - `created` is set after payload fields are written; consumers can only fetch tasks in `created` state.
- **Notification policy**
  - Consumers are woken when at least `batch_size` tasks are ready, or when the buffer is full but there is at least one ready task (to avoid deadlock on a full buffer with a partial batch).
