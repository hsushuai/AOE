package tests;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.file.Paths;

import ai.core.AI;
import ai.jni.Response;
import ai.reward.RewardFunctionInterface;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import rts.GameState;
import rts.PartiallyObservableGameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.Trace;
import rts.TraceEntry;
import rts.units.UnitTypeTable;

/**
 * Instances of this class each let us run a single environment (or sequence
 * of them, if we reset() in between) between two players. 
 * 
 * In this client, it is assumed that actions are selected by Java-based bots 
 * for **both** players. See JNIGridnetClient.java for a client where one
 * player is externally controlled, and JNIGridnetClientSelfPlay.java for one
 * where both players are externally controlled.
 *
 * @author santi and costa
 */
public class JNIBotClient {

    PhysicalGameStateJFrame w;
    public AI ai1;
    public AI ai2;
    PhysicalGameState pgs;
    GameState gs;
    public GameState player1gs, player2gs;
    UnitTypeTable utt;
    boolean partialObs;
    public RewardFunctionInterface[] rfs;
    public String mapPath;
    String micrortsPath;
    boolean gameover = false;
    boolean layerJSON = true;
    public int whiteTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;
    public int blackTheme = PhysicalGameStatePanel.COLORSCHEME_BLACK;

    // storage
    double[] rewards;
    boolean[] dones;
    Response response;
    PlayerAction pa1;
    PlayerAction pa2;
    public Trace trace;

    /**
     * 
     * @param a_rfs Reward functions we want to use to compute rewards at every step.
     * @param a_micrortsPath Path for the microrts root directory (with Java code and maps).
     * @param a_mapPath Path (under microrts root dir) for map to load.
     * @param a_ai1
     * @param a_ai2
     * @param a_utt
     * @param partial_obs
     * @throws Exception
     */
    public JNIBotClient(RewardFunctionInterface[] a_rfs, String a_micrortsPath, String a_mapPath, AI a_ai1, AI a_ai2, UnitTypeTable a_utt, boolean partial_obs) throws Exception{
        micrortsPath = a_micrortsPath;
        mapPath = a_mapPath;
        rfs = a_rfs;
        utt = a_utt;
        partialObs = partial_obs;
        ai1 = a_ai1;
        ai2 = a_ai2;
        if (ai1 == null || ai2 == null) {
            throw new Exception("no ai1 or ai2 was chosen");
        }
        if (micrortsPath.length() != 0) {
            this.mapPath = Paths.get(micrortsPath, mapPath).toString();
        }

        pgs = PhysicalGameState.load(mapPath, utt);
        trace = new Trace(utt);

        // initialize storage
        rewards = new double[rfs.length];
        dones = new boolean[rfs.length];
        response = new Response(null, null, null, null);
    }

    public byte[] render(boolean screenRender, String theme) throws Exception {
        int renderTheme = theme.equals("white") ? whiteTheme : blackTheme;

        if (w == null) {
            w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, partialObs, null, renderTheme);
        }

        w.setStateCloning(gs);

        if (screenRender) {
            w.setVisible(true);
            w.repaint();
        } else {
            w.setVisible(false);
        }

        BufferedImage image = new BufferedImage(640, 640, BufferedImage.TYPE_3BYTE_BGR);
        PhysicalGameStatePanel panel = w.getPanel();
        panel.paint(image.getGraphics());

        WritableRaster raster = image.getRaster();
        DataBufferByte data = (DataBufferByte) raster.getDataBuffer();
        byte [] bd = data.getData();

        if (gs.gameover()) {
            w.dispose();
        }
        
        return bd;
    }

    public Response gameStep(int player) throws Exception {
        if (partialObs) {
            player1gs = new PartiallyObservableGameState(gs, player);
            player2gs = new PartiallyObservableGameState(gs, 1 - player);
        } else {
            player1gs = gs;
            player2gs = gs;
        }

        pa1 = ai1.getAction(player, gs);
        pa2 = ai2.getAction(1 - player, gs);

        gs.issueSafe(pa1);
        gs.issueSafe(pa2);
        
        TraceEntry te  = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
        te.addPlayerAction(pa1.clone());
        te.addPlayerAction(pa2.clone());
        te.addPlayerAction(pa2.clone());
        if (!pa1.isEmpty() || !pa2.isEmpty()) {
            trace.addEntry(te);
        }

        // simulate:
        gameover = gs.cycle();
        if (gameover) {
            ai1.gameOver(gs.winner());
            ai2.gameOver(gs.winner());
        }

        for (int i = 0; i < rewards.length; i++) {
            rfs[i].computeReward(player, 1 - player, te, gs);
            dones[i] = rfs[i].isDone();
            rewards[i] = rfs[i].getReward();
        }
        response.set(
            gs.getVectorObservation(player),
            rewards,
            dones,
            ai1.computeInfo(gs, player1gs));
        return response;
    }

    public Response gameStep() throws Exception {
        int player = 0;
        if (partialObs) {
            player1gs = new PartiallyObservableGameState(gs, player);
            player2gs = new PartiallyObservableGameState(gs, 1 - player);
        } else {
            player1gs = gs;
            player2gs = gs;
        }

        pa1 = ai1.getAction(player, gs);
        pa2 = ai2.getAction(1 - player, gs);

        gs.issueSafe(pa1);
        gs.issueSafe(pa2);
        
        TraceEntry te  = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
        te.addPlayerAction(pa1.clone());
        te.addPlayerAction(pa2.clone());
        te.addPlayerAction(pa2.clone());
        if (!pa1.isEmpty() || !pa2.isEmpty()) {
            trace.addEntry(te);
        }

        // simulate:
        gameover = gs.cycle();
        if (gameover) {
            ai1.gameOver(gs.winner());
            ai2.gameOver(gs.winner());
        }

        for (int i = 0; i < rewards.length; i++) {
            rfs[i].computeReward(player, 1 - player, te, gs);
            dones[i] = rfs[i].isDone();
            rewards[i] = rfs[i].getReward();
        }
        response.set(
            gs.getVectorObservation(player),
            rewards,
            dones,
            ai1.computeInfo(gs, player1gs));
        return response;
    }

    public String sendUTT() throws Exception {
        Writer w = new StringWriter();
        utt.toJSON(w);
        return w.toString(); // now it works fine
    }

    /**
     * Resets the environment.
     * @param player This parameter is unused.
     * @return Response after reset.
     * @throws Exception
     */
    public Response reset(int player) throws Exception {
        if (partialObs) {
            player1gs = new PartiallyObservableGameState(gs, player);
            player2gs = new PartiallyObservableGameState(gs, 1 - player);
        } else {
            player1gs = gs;
            player2gs = gs;
        }
        
        ai1 = ai1.clone();
        ai1.reset();
        ai2 = ai2.clone();
        ai2.reset();
        pgs = PhysicalGameState.load(mapPath, utt);
        gs = new GameState(pgs, utt);

        for (int i = 0; i < rewards.length; i++) {
            rewards[i] = 0;
            dones[i] = false;
        }

        response.set(
            gs.getVectorObservation(player),
            rewards,
            dones,
            ai1.computeInfo(gs, player1gs));
        return response;
    }

    public String getJSONStringTrace() {
        try {
            StringWriter stringWriter = new StringWriter();
            trace.toJSON(stringWriter);
            return stringWriter.toString();
        } catch (Exception e) {
            return null;
        }
    }

    public void close() throws Exception {
        if (w!=null) {
            w.dispose();    
        }
    }
}
