import { useState, useEffect, useRef } from 'react'
import { getJob } from '../services/api'

export function useJobPolling(jobId) {
  const [job, setJob]       = useState(null)
  const [error, setError]   = useState(null)
  const intervalRef         = useRef(null)

  useEffect(() => {
    // Don't start polling if there's no jobId
    if (!jobId) return

    const poll = async () => {
      try {
        const response = await getJob(jobId)
        const jobData  = response.data
        setJob(jobData)

        // Stop polling when job reaches a final state
        if (jobData.status === 'completed' || jobData.status === 'failed') {
          clearInterval(intervalRef.current)
        }

      } catch (err) {
        setError('Could not fetch job status')
        clearInterval(intervalRef.current)
      }
    }

    // Poll immediately then every 3 seconds
    poll()
    intervalRef.current = setInterval(poll, 3000)

    // Cleanup — stop polling when component unmounts
    return () => clearInterval(intervalRef.current)

  }, [jobId])

  return { job, error }
}
import {useState, useEffect, useRef} from 'react'
import { getJob} from ".../services/api"
export function useJobPolling(JobId){
    const [job,setJob] = useState(null)
    const [error, setError] = useState(null)
    const intervalRef = useRef(null)
    useEffect(() => {
        if (!jobId) return
        const poll = async () => {
            try{
                const response = await getJob(jobId)
                const jobData = response.jobData
                setJob(jobData)
                if (jobData.status === "completed" || jobData.status === "failed") {
                    clearInterval(intervalRef.current)
                }
            }catch (err){
                setError("Could not fetch job status")
                clearInterval(intervalRef.current)
            }
        }
        poll()
        intervalRef.current = setInterval(poll, 3000)
        return () => clearInterval(intervalRef.current)
    },[jobId])
    return {job, error}
}

