//! Tests for the UDS module.

#[cfg(test)]
mod tests {
    use crate::uds::uds_codec::{FlowStatus, IsoTpCodec, IsoTpFrame};
    use crate::uds::uds_services::{LoopbackTransport, UdsClient};
    use crate::uds::uds_types::{
        NegativeResponseCode, SessionType, UdsRequest, UdsResponse, UdsServiceId,
    };

    // ---- ISO-TP Single Frame -----------------------------------------------

    #[test]
    fn test_isotp_sf_encode_decode_round_trip() {
        let payload = [0x10u8, 0x03]; // DSC ExtendedDiagnostic
        let encoded = IsoTpFrame::encode_single(&payload).expect("encode SF");
        assert_eq!(encoded[0], 0x02, "SF: type nibble 0 | DL 2");
        assert_eq!(&encoded[1..3], &payload);

        let decoded = IsoTpFrame::decode(&encoded).expect("decode SF");
        match decoded {
            IsoTpFrame::SingleFrame { data_len, data } => {
                assert_eq!(data_len, 2);
                assert_eq!(&data[..2], &payload);
            }
            _ => panic!("expected SingleFrame"),
        }
    }

    #[test]
    fn test_isotp_sf_max_payload() {
        let payload = [0xAAu8; 7];
        let encoded = IsoTpFrame::encode_single(&payload).expect("encode max SF");
        assert_eq!(encoded[0], 0x07);
    }

    #[test]
    fn test_isotp_sf_too_large_rejected() {
        let payload = [0xBBu8; 8];
        assert!(IsoTpFrame::encode_single(&payload).is_err());
    }

    #[test]
    fn test_isotp_ff_encode_decode() {
        let first_6 = [0x22u8, 0xF1, 0x90, 0x00, 0x00, 0x00];
        let encoded = IsoTpFrame::encode_first(20, &first_6).expect("encode FF");
        assert_eq!(encoded[0], 0x10); // type=1, len_hi=0
        assert_eq!(encoded[1], 20); // len_lo=20
        assert_eq!(&encoded[2..8], &first_6);

        let decoded = IsoTpFrame::decode(&encoded).expect("decode FF");
        match decoded {
            IsoTpFrame::FirstFrame { total_len, data } => {
                assert_eq!(total_len, 20);
                assert_eq!(&data[..6], &first_6);
            }
            _ => panic!("expected FirstFrame"),
        }
    }

    #[test]
    fn test_isotp_cf_encode_decode() {
        let seg = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let encoded = IsoTpFrame::encode_consecutive(1, &seg).expect("encode CF");
        assert_eq!(encoded[0], 0x21); // type=2 | seq=1
        assert_eq!(&encoded[1..8], &seg);

        let decoded = IsoTpFrame::decode(&encoded).expect("decode CF");
        match decoded {
            IsoTpFrame::ConsecutiveFrame {
                sequence_number,
                data,
            } => {
                assert_eq!(sequence_number, 1);
                assert_eq!(&data[..7], &seg);
            }
            _ => panic!("expected ConsecutiveFrame"),
        }
    }

    #[test]
    fn test_isotp_flow_control_encode_decode() {
        let encoded = IsoTpFrame::encode_flow_control(FlowStatus::ContinueToSend, 0, 25);
        assert_eq!(encoded[0], 0x30);
        assert_eq!(encoded[1], 0x00);
        assert_eq!(encoded[2], 25);

        let decoded = IsoTpFrame::decode(&encoded).expect("decode FC");
        match decoded {
            IsoTpFrame::FlowControl {
                flow_status,
                block_size,
                st_min,
            } => {
                assert_eq!(flow_status, FlowStatus::ContinueToSend);
                assert_eq!(block_size, 0);
                assert_eq!(st_min, 25);
            }
            _ => panic!("expected FlowControl"),
        }
    }

    // ---- IsoTpCodec reassembly ---------------------------------------------

    #[test]
    fn test_codec_single_frame_reassembly() {
        let mut codec = IsoTpCodec::new();
        let payload = [0x10u8, 0x03]; // DSC request
        let frame = IsoTpFrame::encode_single(&payload).expect("encode");
        let result = codec.feed(&frame).expect("feed");
        assert_eq!(result, Some(vec![0x10, 0x03]));
    }

    #[test]
    fn test_codec_multi_frame_reassembly() {
        let mut codec = IsoTpCodec::new();
        // Build a 13-byte message split across FF + 2×CF
        let msg: Vec<u8> = (0u8..13).collect();

        // First frame: total_len=13, first 6 bytes
        let ff = IsoTpFrame::encode_first(13, &msg[..6]).expect("FF");
        let r1 = codec.feed(&ff).expect("feed FF");
        assert!(r1.is_none(), "not complete after FF");

        // Consecutive frame 1: bytes 6..13 (7 bytes)
        let cf1 = IsoTpFrame::encode_consecutive(1, &msg[6..13]).expect("CF1");
        let r2 = codec.feed(&cf1).expect("feed CF1");
        assert_eq!(r2, Some(msg));
    }

    #[test]
    fn test_codec_segmentation() {
        let mut codec = IsoTpCodec::new();
        // 13 bytes: should produce FF + CF
        let payload: Vec<u8> = (0u8..13).collect();
        codec.segment(&payload).expect("segment");

        let mut frames = Vec::new();
        while let Some(f) = codec.next_frame() {
            frames.push(f);
        }
        assert_eq!(frames.len(), 2, "FF + 1 CF");

        // Decode them back
        let mut rx = IsoTpCodec::new();
        let r1 = rx.feed(&frames[0]).expect("feed FF");
        assert!(r1.is_none());
        let r2 = rx.feed(&frames[1]).expect("feed CF");
        assert_eq!(r2.as_deref(), Some(payload.as_slice()));
    }

    // ---- UDS Request / Response --------------------------------------------

    #[test]
    fn test_uds_request_encode_decode_dsc() {
        let req = UdsRequest::new(UdsServiceId::DiagnosticSessionControl)
            .with_sub_function(SessionType::ExtendedDiagnostic as u8);
        let encoded = req.encode();
        assert_eq!(encoded, vec![0x10, 0x03]);

        let decoded = UdsRequest::decode(&encoded, true).expect("decode DSC");
        assert_eq!(decoded.service_id, UdsServiceId::DiagnosticSessionControl);
        assert_eq!(decoded.sub_function, Some(0x03));
    }

    #[test]
    fn test_uds_request_encode_decode_rdbi() {
        let req = UdsRequest::new(UdsServiceId::ReadDataByIdentifier).with_data(vec![0xF1, 0x90]); // VIN data ID
        let encoded = req.encode();
        assert_eq!(encoded, vec![0x22, 0xF1, 0x90]);

        let decoded = UdsRequest::decode(&encoded, false).expect("decode RDBI");
        assert_eq!(decoded.service_id, UdsServiceId::ReadDataByIdentifier);
        assert_eq!(decoded.data, vec![0xF1, 0x90]);
    }

    #[test]
    fn test_uds_response_positive_decode() {
        // Positive DSC response
        let raw = vec![0x50u8, 0x03, 0x00, 0x19, 0x01, 0xF4];
        let resp = UdsResponse::decode(&raw).expect("decode positive DSC resp");
        match resp {
            UdsResponse::Positive {
                service_id,
                sub_function,
                data,
            } => {
                assert_eq!(service_id, UdsServiceId::DiagnosticSessionControl);
                assert_eq!(sub_function, Some(0x03));
                assert_eq!(data, vec![0x00, 0x19, 0x01, 0xF4]);
            }
            _ => panic!("expected positive"),
        }
    }

    #[test]
    fn test_uds_response_negative_decode() {
        let raw = vec![0x7Fu8, 0x22, 0x31]; // NRC RequestOutOfRange for RDBI
        let resp = UdsResponse::decode(&raw).expect("decode negative");
        match resp {
            UdsResponse::Negative { service_id, nrc } => {
                assert_eq!(service_id, UdsServiceId::ReadDataByIdentifier);
                assert_eq!(nrc, NegativeResponseCode::RequestOutOfRange);
            }
            _ => panic!("expected negative"),
        }
    }

    #[test]
    fn test_uds_response_encode_negative() {
        let resp = UdsResponse::Negative {
            service_id: UdsServiceId::SecurityAccess,
            nrc: NegativeResponseCode::InvalidKey,
        };
        let encoded = resp.encode();
        assert_eq!(encoded, vec![0x7F, 0x27, 0x35]);
    }

    // ---- NRC codes ---------------------------------------------------------

    #[test]
    fn test_nrc_round_trip_all_known() {
        let known: &[(u8, NegativeResponseCode)] = &[
            (0x10, NegativeResponseCode::GeneralReject),
            (0x11, NegativeResponseCode::ServiceNotSupported),
            (0x12, NegativeResponseCode::SubFunctionNotSupported),
            (
                0x13,
                NegativeResponseCode::IncorrectMessageLengthOrInvalidFormat,
            ),
            (0x22, NegativeResponseCode::ConditionsNotCorrect),
            (0x31, NegativeResponseCode::RequestOutOfRange),
            (0x33, NegativeResponseCode::SecurityAccessDenied),
            (0x35, NegativeResponseCode::InvalidKey),
            (
                0x78,
                NegativeResponseCode::RequestCorrectlyReceivedResponsePending,
            ),
            (
                0x7F,
                NegativeResponseCode::ServiceNotSupportedInActiveSession,
            ),
        ];
        for &(raw, expected) in known {
            let got = NegativeResponseCode::from_byte(raw);
            assert_eq!(got, expected, "NRC 0x{:02X}", raw);
            assert_eq!(got as u8, raw);
        }
    }

    #[test]
    fn test_nrc_description_non_empty() {
        for raw in 0u8..=0xFFu8 {
            let nrc = NegativeResponseCode::from_byte(raw);
            assert!(!nrc.description().is_empty());
        }
    }

    // ---- Service ID helpers ------------------------------------------------

    #[test]
    fn test_service_id_positive_response_ids() {
        assert_eq!(
            UdsServiceId::ReadDataByIdentifier.positive_response_id(),
            0x62
        );
        assert_eq!(
            UdsServiceId::DiagnosticSessionControl.positive_response_id(),
            0x50
        );
        assert_eq!(UdsServiceId::SecurityAccess.positive_response_id(), 0x67);
    }

    #[test]
    fn test_service_id_roundtrip() {
        let ids: &[u8] = &[
            0x10, 0x11, 0x14, 0x19, 0x22, 0x27, 0x28, 0x2E, 0x2F, 0x31, 0x34, 0x36, 0x37, 0x3E,
            0x7F,
        ];
        for &id in ids {
            let sid = UdsServiceId::from_byte(id);
            assert!(sid.is_some(), "SID 0x{:02X} should parse", id);
            assert_eq!(sid.expect("SID should parse") as u8, id);
        }
    }

    // ---- UDS Client loopback -----------------------------------------------

    #[test]
    fn test_uds_client_rdbi_loopback() {
        let transport = LoopbackTransport::new();

        // Pre-inject a positive RDBI response for DID 0xF190 (VIN):
        // Positive: [0x62, 0xF1, 0x90, 0x57, 0x30, 0x52]
        // This fits in a single ISO-TP frame: [0x06, 0x62, 0xF1, 0x90, 0x57, 0x30, 0x52, 0x00]
        let resp_payload = vec![0x62u8, 0xF1, 0x90, 0x57, 0x30, 0x52];
        let isotp_frame = IsoTpFrame::encode_single(&resp_payload).expect("encode resp");

        // We can't use async inject in a sync test; use try_lock approach
        {
            let mut q = transport.queue.try_lock().expect("lock");
            q.push_back(isotp_frame);
        }

        let mut client = UdsClient::new(transport, 0x7DF, 0x7E8);
        let record = client.read_data_by_id(0xF190).expect("RDBI");
        assert_eq!(record, vec![0x57, 0x30, 0x52]);
    }

    #[test]
    fn test_uds_client_security_access_loopback() {
        let transport = LoopbackTransport::new();

        // Inject seed response: [0x67, 0x01, 0xAA, 0xBB]
        let seed_resp = vec![0x67u8, 0x01, 0xAA, 0xBB];
        let seed_frame = IsoTpFrame::encode_single(&seed_resp).expect("encode seed resp");
        {
            let mut q = transport.queue.try_lock().expect("lock");
            q.push_back(seed_frame);
        }

        let mut client = UdsClient::new(transport, 0x7DF, 0x7E8);
        let seed = client.security_access_seed(0x01).expect("seed");
        // Sub-function is consumed; data is the remaining bytes after sf
        // Response: [0x67, sub_fn=0x01, 0xAA, 0xBB] → Positive { sf=0x01, data=[0xAA, 0xBB] }
        assert_eq!(seed, vec![0xAA, 0xBB]);
    }

    #[test]
    fn test_uds_client_negative_response() {
        let transport = LoopbackTransport::new();

        // Inject NRC SecurityAccessDenied
        let nrc_resp = vec![0x7Fu8, 0x27, 0x33];
        let nrc_frame = IsoTpFrame::encode_single(&nrc_resp).expect("encode nrc");
        {
            let mut q = transport.queue.try_lock().expect("lock");
            q.push_back(nrc_frame);
        }

        let mut client = UdsClient::new(transport, 0x7DF, 0x7E8);
        let result = client.security_access_seed(0x01);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("rejected") || err_msg.contains("denied") || err_msg.contains("NRC"),
            "error should mention rejection: {}",
            err_msg
        );
    }
}
