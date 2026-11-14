import Flutter
import UIKit
import AVFoundation

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private var acousticSession: AcousticMeasurementSession?
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    if let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "AcousticMeasurementSession") {
      acousticSession = AcousticMeasurementSession(messenger: registrar.messenger())
    }
  }
}

// record 7.2 exposes category/activation but no AVAudioSession mode setting.
// It remains responsible for permission, stream format and interruption policy.
private final class AcousticMeasurementSession {
  private let channel: FlutterMethodChannel
  private var previous: (AVAudioSession.Category, AVAudioSession.Mode, AVAudioSession.CategoryOptions)?
  private var invalidated = false
  private var observers: [NSObjectProtocol] = []

  init(messenger: FlutterBinaryMessenger) {
    channel = FlutterMethodChannel(name: "thud/acoustic_audio_session", binaryMessenger: messenger)
    channel.setMethodCallHandler { [weak self] call, result in
      guard let self = self else { return result(FlutterError(code: "closed", message: "Audio session closed", details: nil)) }
      do {
        switch call.method {
        case "begin": try self.begin(); result(nil)
        case "assertValid":
          guard self.previous != nil, !self.invalidated else {
            return result(FlutterError(code: "interrupted", message: "Recording interrupted or audio input changed. Record again.", details: nil))
          }
          result(nil)
        case "end": try self.end(); result(nil)
        default: result(FlutterMethodNotImplemented)
        }
      } catch {
        result(FlutterError(code: "audio_session", message: error.localizedDescription, details: nil))
      }
    }
  }

  private func begin() throws {
    guard previous == nil else { return }
    let session = AVAudioSession.sharedInstance()
    // Dart/record requests permission first; never prompt a second time here.
    guard session.recordPermission == .granted else {
      throw NSError(domain: "THUD", code: 1, userInfo: [NSLocalizedDescriptionKey: "Microphone permission is required."])
    }
    previous = (session.category, session.mode, session.categoryOptions)
    invalidated = false
    do {
      try session.setCategory(.record, mode: .measurement, options: [])
      try session.setPreferredSampleRate(48000)
      try session.setActive(true)
      try session.setPreferredInputNumberOfChannels(1)
      let center = NotificationCenter.default
      for name in [AVAudioSession.interruptionNotification, AVAudioSession.routeChangeNotification,
                   AVAudioSession.mediaServicesWereResetNotification, UIApplication.didEnterBackgroundNotification] {
        observers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
          // Never resume a partly interrupted measurement or join two routes.
          self?.invalidated = true
        })
      }
    } catch {
      try? end()
      throw error
    }
  }

  private func end() throws {
    for observer in observers { NotificationCenter.default.removeObserver(observer) }
    observers.removeAll()
    guard let old = previous else { return }
    previous = nil
    let session = AVAudioSession.sharedInstance()
    // Restore category even if deactivation fails (e.g. a system interruption).
    var failure: Error?
    do { try session.setActive(false, options: .notifyOthersOnDeactivation) } catch { failure = error }
    do { try session.setCategory(old.0, mode: old.1, options: old.2) } catch { if failure == nil { failure = error } }
    if let failure = failure { throw failure }
  }

  deinit {
    channel.setMethodCallHandler(nil)
    try? end()
  }
}
